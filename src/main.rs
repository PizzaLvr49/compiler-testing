use ariadne::{Color, Label, Report, ReportKind, Source};
use chumsky::prelude::*;
use cranelift::jit::{JITBuilder, JITModule};
use cranelift::module::{FuncId, Linkage, Module, default_libcall_names};
use cranelift::prelude::*;
use std::collections::HashMap;

#[derive(Debug)]
enum Expr<'src> {
    Num(f64),
    Var(&'src str),

    Neg(Box<Self>),
    Add(Box<Self>, Box<Self>),
    Sub(Box<Self>, Box<Self>),
    Mul(Box<Self>, Box<Self>),
    Div(Box<Self>, Box<Self>),

    Call(&'src str, Vec<Self>),

    Let {
        name: &'src str,
        rhs: Box<Self>,
        then: Box<Self>,
    },

    Fn {
        name: &'src str,
        args: Vec<&'src str>,
        body: Box<Self>,
        then: Box<Self>,
    },
}

#[expect(clippy::let_and_return)]
fn parser<'src>() -> impl Parser<'src, &'src str, Expr<'src>, extra::Err<Rich<'src, char>>> {
    let ident = text::ascii::ident().padded();

    let expr = recursive(|expr| {
        let int = text::int(10).map(|s: &str| Expr::Num(s.parse().unwrap()));

        let call = ident
            .then(
                expr.clone()
                    .separated_by(just(',').padded())
                    .allow_trailing()
                    .collect::<Vec<_>>()
                    .delimited_by(just('(').padded(), just(')').padded()),
            )
            .map(|(f, args)| Expr::Call(f, args));

        let atom = choice((
            int,
            expr.clone()
                .delimited_by(just('(').padded(), just(')').padded()),
            call,
            ident.map(Expr::Var),
        ))
        .padded();

        let op = |c| just(c).padded();

        let unary = op('-')
            .repeated()
            .foldr(atom, |_op, rhs| Expr::Neg(Box::new(rhs)));

        let product = unary.clone().foldl(
            choice((
                op('*').to(Expr::Mul as fn(_, _) -> _),
                op('/').to(Expr::Div as fn(_, _) -> _),
            ))
            .then(unary)
            .repeated(),
            |lhs, (op, rhs)| op(Box::new(lhs), Box::new(rhs)),
        );

        let sum = product.clone().foldl(
            choice((
                op('+').to(Expr::Add as fn(_, _) -> _),
                op('-').to(Expr::Sub as fn(_, _) -> _),
            ))
            .then(product)
            .repeated(),
            |lhs, (op, rhs)| op(Box::new(lhs), Box::new(rhs)),
        );

        sum
    });

    let decl = recursive(|decl| {
        let r#let = text::ascii::keyword("let")
            .padded()
            .ignore_then(ident)
            .then_ignore(just('=').padded())
            .then(expr.clone())
            .then_ignore(just(';').padded())
            .then(decl.clone())
            .map(|((name, rhs), then)| Expr::Let {
                name,
                rhs: Box::new(rhs),
                then: Box::new(then),
            });

        let r#fn = text::ascii::keyword("fn")
            .padded()
            .ignore_then(ident)
            .then(ident.repeated().collect::<Vec<_>>())
            .then_ignore(just('=').padded())
            .then(expr.clone())
            .then_ignore(just(';').padded())
            .then(decl)
            .map(|(((name, args), body), then)| Expr::Fn {
                name,
                args,
                body: Box::new(body),
                then: Box::new(then),
            });

        choice((r#let, r#fn, expr)).padded()
    });

    decl.then_ignore(end())
}

fn validate<'src>(
    expr: &'src Expr<'src>,
    vars: &mut Vec<&'src str>,
    funcs: &mut HashMap<&'src str, usize>,
) -> Result<(), String> {
    match expr {
        Expr::Num(_) => Ok(()),
        Expr::Var(name) => {
            if vars.contains(name) {
                Ok(())
            } else {
                Err(format!("Cannot find variable `{name}` in scope"))
            }
        }
        Expr::Neg(a) => validate(a, vars, funcs),
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) => {
            validate(a, vars, funcs)?;
            validate(b, vars, funcs)
        }
        Expr::Let { name, rhs, then } => {
            validate(rhs, vars, funcs)?;
            vars.push(*name);
            let res = validate(then, vars, funcs);
            vars.pop();
            res
        }
        Expr::Fn {
            name,
            args,
            body,
            then,
        } => {
            funcs.insert(*name, args.len());

            let mut body_vars = args.clone();
            validate(body, &mut body_vars, funcs)?;

            validate(then, vars, funcs)
        }
        Expr::Call(name, args) => {
            if let Some(&expected_arity) = funcs.get(name) {
                if args.len() != expected_arity {
                    return Err(format!(
                        "Wrong number of arguments for function `{name}`: expected {expected_arity}, found {}",
                        args.len()
                    ));
                }
                for arg in args {
                    validate(arg, vars, funcs)?;
                }
                Ok(())
            } else {
                Err(format!("Cannot find function `{name}` in scope"))
            }
        }
    }
}

struct JITCompiler {
    builder_context: FunctionBuilderContext,
    ctx: codegen::Context,
    module: JITModule,
    funcs: HashMap<String, FuncId>,
}

impl JITCompiler {
    fn new() -> Self {
        let mut flag_builder = settings::builder();
        flag_builder.set("use_colocated_libcalls", "false").unwrap();
        flag_builder.set("is_pic", "false").unwrap();

        let isa_builder = cranelift::native::builder().unwrap();
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .unwrap();

        let builder = JITBuilder::with_isa(isa, default_libcall_names());
        let module = JITModule::new(builder);

        Self {
            builder_context: FunctionBuilderContext::new(),
            ctx: module.make_context(),
            module,
            funcs: HashMap::new(),
        }
    }

    fn collect_and_declare_functions(&mut self, expr: &Expr) {
        match expr {
            Expr::Fn {
                name,
                args,
                body,
                then,
            } => {
                let mut sig = self.module.make_signature();
                for _ in args {
                    sig.params.push(AbiParam::new(types::F64));
                }
                sig.returns.push(AbiParam::new(types::F64));

                let id = self
                    .module
                    .declare_function(name, Linkage::Local, &sig)
                    .unwrap();
                self.funcs.insert((*name).to_string(), id);

                self.collect_and_declare_functions(body);
                self.collect_and_declare_functions(then);
            }
            Expr::Let { rhs, then, .. } => {
                self.collect_and_declare_functions(rhs);
                self.collect_and_declare_functions(then);
            }
            Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) => {
                self.collect_and_declare_functions(a);
                self.collect_and_declare_functions(b);
            }
            Expr::Neg(a) => self.collect_and_declare_functions(a),
            Expr::Call(_, args) => {
                for arg in args {
                    self.collect_and_declare_functions(arg);
                }
            }
            Expr::Num(_) | Expr::Var(_) => {}
        }
    }

    fn compile_defined_functions<'src>(&mut self, expr: &'src Expr<'src>) {
        match expr {
            Expr::Fn {
                name,
                args,
                body,
                then,
            } => {
                self.ctx.clear();
                for _ in args {
                    self.ctx
                        .func
                        .signature
                        .params
                        .push(AbiParam::new(types::F64));
                }
                self.ctx
                    .func
                    .signature
                    .returns
                    .push(AbiParam::new(types::F64));

                let mut builder =
                    FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_context);
                let entry_block = builder.create_block();
                builder.append_block_params_for_function_params(entry_block);
                builder.switch_to_block(entry_block);
                builder.seal_block(entry_block);

                let mut vars = HashMap::new();

                for (i, arg) in args.iter().enumerate() {
                    let val = builder.block_params(entry_block)[i];
                    let var = builder.declare_var(types::F64);
                    builder.def_var(var, val);
                    vars.insert(*arg, var);
                }

                let mut translator = Translator {
                    builder,
                    module: &mut self.module,
                    funcs: &self.funcs,
                    vars,
                };

                let ret = translator.translate(body);
                translator.builder.ins().return_(&[ret]);
                translator.builder.finalize();

                let id = *self.funcs.get(*name).unwrap();
                self.module.define_function(id, &mut self.ctx).unwrap();
                self.module.clear_context(&mut self.ctx);

                self.compile_defined_functions(body);
                self.compile_defined_functions(then);
            }
            Expr::Let { rhs, then, .. } => {
                self.compile_defined_functions(rhs);
                self.compile_defined_functions(then);
            }
            Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) => {
                self.compile_defined_functions(a);
                self.compile_defined_functions(b);
            }
            Expr::Neg(a) => self.compile_defined_functions(a),
            Expr::Call(_, args) => {
                for arg in args {
                    self.compile_defined_functions(arg);
                }
            }
            Expr::Num(_) | Expr::Var(_) => {}
        }
    }

    fn run_main(&mut self, ast: &Expr) -> f64 {
        self.ctx.clear();
        self.ctx
            .func
            .signature
            .returns
            .push(AbiParam::new(types::F64));

        let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_context);
        let entry_block = builder.create_block();
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);

        let mut translator = Translator {
            builder,
            module: &mut self.module,
            funcs: &self.funcs,
            vars: HashMap::new(),
        };

        let ret = translator.translate(ast);
        translator.builder.ins().return_(&[ret]);
        translator.builder.finalize();

        let mut sig = self.module.make_signature();
        sig.returns.push(AbiParam::new(types::F64));

        let id = self
            .module
            .declare_function("__main", Linkage::Export, &sig)
            .unwrap();
        self.module.define_function(id, &mut self.ctx).unwrap();
        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions().unwrap();

        let code = self.module.get_finalized_function(id);
        let func: fn() -> f64 = unsafe { std::mem::transmute(code) };
        func()
    }
}

struct Translator<'a, 'm> {
    builder: FunctionBuilder<'a>,
    module: &'m mut JITModule,
    funcs: &'m HashMap<String, FuncId>,
    vars: HashMap<&'a str, Variable>,
}

impl<'a> Translator<'a, '_> {
    fn translate(&mut self, expr: &'a Expr<'a>) -> Value {
        match expr {
            Expr::Num(n) => self.builder.ins().f64const(*n),
            Expr::Var(name) => {
                let var = self.vars.get(name).unwrap();
                self.builder.use_var(*var)
            }
            Expr::Neg(a) => {
                let val = self.translate(a);
                self.builder.ins().fneg(val)
            }
            Expr::Add(a, b) => {
                let av = self.translate(a);
                let bv = self.translate(b);
                self.builder.ins().fadd(av, bv)
            }
            Expr::Sub(a, b) => {
                let av = self.translate(a);
                let bv = self.translate(b);
                self.builder.ins().fsub(av, bv)
            }
            Expr::Mul(a, b) => {
                let av = self.translate(a);
                let bv = self.translate(b);
                self.builder.ins().fmul(av, bv)
            }
            Expr::Div(a, b) => {
                let av = self.translate(a);
                let bv = self.translate(b);
                self.builder.ins().fdiv(av, bv)
            }
            Expr::Let { name, rhs, then } => {
                let rhs_val = self.translate(rhs);
                let var = self.builder.declare_var(types::F64);
                self.builder.def_var(var, rhs_val);

                let old_var = self.vars.insert(*name, var);
                let res = self.translate(then);

                if let Some(old) = old_var {
                    self.vars.insert(*name, old);
                } else {
                    self.vars.remove(*name);
                }
                res
            }
            Expr::Call(name, args) => {
                let id = self.funcs.get(*name).unwrap();
                let local_callee = self.module.declare_func_in_func(*id, self.builder.func);

                let mut arg_vals = Vec::new();
                for arg in args {
                    arg_vals.push(self.translate(arg));
                }

                let call = self.builder.ins().call(local_callee, &arg_vals);
                self.builder.inst_results(call)[0]
            }
            Expr::Fn { then, .. } => self.translate(then),
        }
    }
}

fn codegen(ast: &Expr) -> Result<f64, String> {
    validate(ast, &mut Vec::new(), &mut HashMap::new())?;

    let mut jit = JITCompiler::new();
    jit.collect_and_declare_functions(ast);
    jit.compile_defined_functions(ast);
    Ok(jit.run_main(ast))
}

fn main() {
    let src = std::fs::read_to_string(std::env::args().nth(1).expect("expected file argument"))
        .expect("failed to read file");

    let (ast, errs) = parser().parse(&src).into_output_errors();

    for e in errs {
        Report::build(ReportKind::Error, ((), e.span().into_range()))
            .with_config(ariadne::Config::new().with_index_type(ariadne::IndexType::Byte))
            .with_message(e.to_string())
            .with_label(
                Label::new(((), e.span().into_range()))
                    .with_message(e.reason().to_string())
                    .with_color(Color::Red),
            )
            .finish()
            .print(Source::from(&src))
            .unwrap();
    }

    if let Some(ast) = ast {
        match codegen(&ast) {
            Ok(output) => println!("{output}"),
            Err(eval_err) => println!("Evaluation error: {eval_err}"),
        }
    }
}
