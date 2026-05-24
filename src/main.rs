use ariadne::{Color, Label, Report, ReportKind, Source};
use chumsky::prelude::*;
use clap::Parser as ClapParser;
use cranelift::jit::{JITBuilder, JITModule};
use cranelift::module::{FuncId, Linkage, Module, default_libcall_names};
use cranelift::prelude::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use sysexits::ExitCode;
use thiserror::Error;

slotmap::new_key_type! {
    struct NodeId;
}

#[derive(Debug)]
enum Node<'src> {
    Num(f64),
    Var(&'src str),
    Neg(NodeId),
    Add(NodeId, NodeId),
    Sub(NodeId, NodeId),
    Mul(NodeId, NodeId),
    Div(NodeId, NodeId),
    Call(&'src str, Vec<NodeId>),
    Let {
        name: &'src str,
        rhs: NodeId,
        then: NodeId,
    },
    Fn {
        name: &'src str,
        args: Vec<&'src str>,
        body: NodeId,
        then: NodeId,
    },
}

#[derive(Error, Debug)]
enum CompilerError {
    #[error("Variable `{0}` not found in scope")]
    VariableNotFound(String),

    #[error("Function `{0}` not found in scope")]
    FunctionNotFound(String),

    #[error("Wrong number of arguments for function `{name}`: expected {expected}, found {found}")]
    ArgumentCountMismatch {
        name: String,
        expected: usize,
        found: usize,
    },

    #[error("Redefinition of function `{0}` in the same scope")]
    DuplicateFunction(String),

    #[error("JIT Compilation Error: {0}")]
    JitError(String),
}

#[expect(clippy::let_and_return)]
fn parser<'src, 'a>(
    arena: &'a RefCell<slotmap::SlotMap<NodeId, Node<'src>>>,
) -> impl Parser<'src, &'src str, NodeId, extra::Err<Rich<'src, char>>> + 'a {
    let alloc = move |node| arena.borrow_mut().insert(node);

    let ident = text::ascii::ident().padded();

    let expr = recursive(|expr| {
        let int = text::int(10).map(move |s: &str| alloc(Node::Num(s.parse().unwrap())));

        let call = ident
            .then(
                expr.clone()
                    .separated_by(just(',').padded())
                    .allow_trailing()
                    .collect::<Vec<_>>()
                    .delimited_by(just('(').padded(), just(')').padded()),
            )
            .map(move |(f, args)| alloc(Node::Call(f, args)));

        let atom = choice((
            int,
            expr.clone()
                .delimited_by(just('(').padded(), just(')').padded()),
            call,
            ident.map(move |s| alloc(Node::Var(s))),
        ))
        .padded();

        let op = |c| just(c).padded();

        let unary = op('-')
            .repeated()
            .foldr(atom, move |_op, rhs| alloc(Node::Neg(rhs)));

        let product = unary.clone().foldl(
            choice((op('*').to(0), op('/').to(1)))
                .then(unary)
                .repeated(),
            move |lhs, (op, rhs)| {
                if op == 0 {
                    alloc(Node::Mul(lhs, rhs))
                } else {
                    alloc(Node::Div(lhs, rhs))
                }
            },
        );

        let sum = product.clone().foldl(
            choice((op('+').to(0), op('-').to(1)))
                .then(product)
                .repeated(),
            move |lhs, (op, rhs)| {
                if op == 0 {
                    alloc(Node::Add(lhs, rhs))
                } else {
                    alloc(Node::Sub(lhs, rhs))
                }
            },
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
            .map(move |((name, rhs), then)| alloc(Node::Let { name, rhs, then }));

        let r#fn = text::ascii::keyword("fn")
            .padded()
            .ignore_then(ident)
            .then(ident.repeated().collect::<Vec<_>>())
            .then_ignore(just('=').padded())
            .then(expr.clone())
            .then_ignore(just(';').padded())
            .then(decl)
            .map(move |(((name, args), body), then)| {
                alloc(Node::Fn {
                    name,
                    args,
                    body,
                    then,
                })
            });

        choice((r#let, r#fn, expr)).padded()
    });

    decl.then_ignore(end())
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

    fn pre_declare(
        &mut self,
        id: NodeId,
        nodes: &slotmap::SlotMap<NodeId, Node<'_>>,
        func_arities: &mut HashMap<String, usize>,
    ) -> Result<(), CompilerError> {
        match &nodes[id] {
            Node::Fn {
                name,
                args,
                body,
                then,
            } => {
                if func_arities.contains_key(*name) {
                    return Err(CompilerError::DuplicateFunction((*name).to_string()));
                }
                func_arities.insert((*name).to_string(), args.len());

                let mut sig = self.module.make_signature();
                for _ in args {
                    sig.params.push(AbiParam::new(types::F64));
                }
                sig.returns.push(AbiParam::new(types::F64));

                let func_id = self
                    .module
                    .declare_function(name, Linkage::Local, &sig)
                    .map_err(|e| CompilerError::JitError(e.to_string()))?;
                self.funcs.insert((*name).to_string(), func_id);

                self.pre_declare(*body, nodes, func_arities)?;
                self.pre_declare(*then, nodes, func_arities)?;
            }
            Node::Let { rhs, then, .. } => {
                self.pre_declare(*rhs, nodes, func_arities)?;
                self.pre_declare(*then, nodes, func_arities)?;
            }
            Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b) | Node::Div(a, b) => {
                self.pre_declare(*a, nodes, func_arities)?;
                self.pre_declare(*b, nodes, func_arities)?;
            }
            Node::Neg(a) => self.pre_declare(*a, nodes, func_arities)?,
            Node::Call(_, args) => {
                for &arg in args {
                    self.pre_declare(arg, nodes, func_arities)?;
                }
            }
            Node::Num(_) | Node::Var(_) => {}
        }
        Ok(())
    }

    fn compile_functions(
        &mut self,
        id: NodeId,
        nodes: &slotmap::SlotMap<NodeId, Node<'_>>,
        func_arities: &HashMap<String, usize>,
    ) -> Result<(), CompilerError> {
        match &nodes[id] {
            Node::Fn {
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
                    func_arities,
                    nodes,
                    vars,
                };

                let ret = translator.translate(*body)?;
                translator.builder.ins().return_(&[ret]);
                translator.builder.finalize();

                let func_id = *self.funcs.get(*name).unwrap();
                self.module
                    .define_function(func_id, &mut self.ctx)
                    .map_err(|e| CompilerError::JitError(e.to_string()))?;
                self.module.clear_context(&mut self.ctx);

                self.compile_functions(*body, nodes, func_arities)?;
                self.compile_functions(*then, nodes, func_arities)?;
            }
            Node::Let { rhs, then, .. } => {
                self.compile_functions(*rhs, nodes, func_arities)?;
                self.compile_functions(*then, nodes, func_arities)?;
            }
            Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b) | Node::Div(a, b) => {
                self.compile_functions(*a, nodes, func_arities)?;
                self.compile_functions(*b, nodes, func_arities)?;
            }
            Node::Neg(a) => self.compile_functions(*a, nodes, func_arities)?,
            Node::Call(_, args) => {
                for &arg in args {
                    self.compile_functions(arg, nodes, func_arities)?;
                }
            }
            Node::Num(_) | Node::Var(_) => {}
        }
        Ok(())
    }

    fn run_main(
        &mut self,
        root: NodeId,
        nodes: &slotmap::SlotMap<NodeId, Node<'_>>,
        func_arities: &HashMap<String, usize>,
    ) -> Result<f64, CompilerError> {
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
            func_arities,
            nodes,
            vars: HashMap::new(),
        };

        let ret = translator.translate(root)?;
        translator.builder.ins().return_(&[ret]);
        translator.builder.finalize();

        let mut sig = self.module.make_signature();
        sig.returns.push(AbiParam::new(types::F64));

        let id = self
            .module
            .declare_function("__main", Linkage::Export, &sig)
            .map_err(|e| CompilerError::JitError(e.to_string()))?;
        self.module
            .define_function(id, &mut self.ctx)
            .map_err(|e| CompilerError::JitError(e.to_string()))?;
        self.module.clear_context(&mut self.ctx);

        self.module
            .finalize_definitions()
            .map_err(|e| CompilerError::JitError(e.to_string()))?;

        let code = self.module.get_finalized_function(id);
        let func: fn() -> f64 = unsafe { std::mem::transmute(code) };
        Ok(func())
    }
}

struct Translator<'a, 'm, 'src> {
    builder: FunctionBuilder<'a>,
    module: &'m mut JITModule,
    funcs: &'m HashMap<String, FuncId>,
    func_arities: &'m HashMap<String, usize>,
    nodes: &'src slotmap::SlotMap<NodeId, Node<'src>>,
    vars: HashMap<&'src str, Variable>,
}

impl Translator<'_, '_, '_> {
    fn translate(&mut self, id: NodeId) -> Result<Value, CompilerError> {
        match &self.nodes[id] {
            Node::Num(n) => Ok(self.builder.ins().f64const(*n)),
            Node::Var(name) => {
                let var = self
                    .vars
                    .get(name)
                    .ok_or_else(|| CompilerError::VariableNotFound((*name).to_string()))?;
                Ok(self.builder.use_var(*var))
            }
            Node::Neg(a) => {
                let val = self.translate(*a)?;
                Ok(self.builder.ins().fneg(val))
            }
            Node::Add(a, b) => {
                let av = self.translate(*a)?;
                let bv = self.translate(*b)?;
                Ok(self.builder.ins().fadd(av, bv))
            }
            Node::Sub(a, b) => {
                let av = self.translate(*a)?;
                let bv = self.translate(*b)?;
                Ok(self.builder.ins().fsub(av, bv))
            }
            Node::Mul(a, b) => {
                let av = self.translate(*a)?;
                let bv = self.translate(*b)?;
                Ok(self.builder.ins().fmul(av, bv))
            }
            Node::Div(a, b) => {
                let av = self.translate(*a)?;
                let bv = self.translate(*b)?;
                Ok(self.builder.ins().fdiv(av, bv))
            }
            Node::Let { name, rhs, then } => {
                let rhs_val = self.translate(*rhs)?;
                let var = self.builder.declare_var(types::F64);
                self.builder.def_var(var, rhs_val);

                let old_var = self.vars.insert(name, var);
                let res = self.translate(*then);

                if let Some(old) = old_var {
                    self.vars.insert(name, old);
                } else {
                    self.vars.remove(name);
                }
                res
            }
            Node::Call(name, args) => {
                let expected_arity = self
                    .func_arities
                    .get(*name)
                    .ok_or_else(|| CompilerError::FunctionNotFound((name).to_string()))?;

                if args.len() != *expected_arity {
                    return Err(CompilerError::ArgumentCountMismatch {
                        name: (*name).to_string(),
                        expected: *expected_arity,
                        found: args.len(),
                    });
                }

                let func_id = self.funcs.get(*name).unwrap();
                let local_callee = self
                    .module
                    .declare_func_in_func(*func_id, self.builder.func);

                let mut arg_vals = Vec::new();
                for arg in args {
                    arg_vals.push(self.translate(*arg)?);
                }

                let call = self.builder.ins().call(local_callee, &arg_vals);
                Ok(self.builder.inst_results(call)[0])
            }
            Node::Fn { then, .. } => self.translate(*then),
        }
    }
}

fn codegen(root: NodeId, nodes: &slotmap::SlotMap<NodeId, Node<'_>>) -> Result<f64, CompilerError> {
    let mut jit = JITCompiler::new();
    let mut func_arities = HashMap::new();

    jit.pre_declare(root, nodes, &mut func_arities)?;
    jit.compile_functions(root, nodes, &func_arities)?;
    jit.run_main(root, nodes, &func_arities)
}

#[derive(ClapParser, Debug)]
#[command(name = "foo-compiler", version, about, long_about = None)]
struct Args {
    #[arg(value_name = "FILE")]
    file: PathBuf,
}

fn main() -> ExitCode {
    let args = match Args::try_parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::Usage;
        }
    };

    let file_path_str = args.file.to_string_lossy().into_owned();

    let src = match std::fs::read_to_string(&args.file) {
        Ok(content) => content,
        Err(err) => {
            eprintln!(
                "Error: Failed to read file '{}': {err}",
                args.file.display()
            );
            match err.kind() {
                std::io::ErrorKind::NotFound | std::io::ErrorKind::PermissionDenied => {
                    return sysexits::ExitCode::NoInput;
                }
                _ => {
                    return sysexits::ExitCode::IoErr;
                }
            }
        }
    };

    let arena = RefCell::new(slotmap::SlotMap::<NodeId, _>::with_key());
    let (ast, errs) = parser(&arena).parse(&src).into_output_errors();

    for e in errs {
        Report::build(
            ReportKind::Error,
            (file_path_str.clone(), e.span().into_range()),
        )
        .with_config(ariadne::Config::new().with_index_type(ariadne::IndexType::Char))
        .with_message(e.to_string())
        .with_label(
            Label::new((file_path_str.clone(), e.span().into_range()))
                .with_message(e.reason().to_string())
                .with_color(Color::Red),
        )
        .finish()
        .eprint((file_path_str.clone(), Source::from(src.as_str())))
        .unwrap();
    }

    if let Some(root_id) = ast {
        let nodes = arena.into_inner();
        match codegen(root_id, &nodes) {
            Ok(output) => println!("{output}"),
            Err(compiler_err) => {
                eprintln!("Compilation failed: {compiler_err}");
                return ExitCode::Software;
            }
        }
    }
    ExitCode::Ok
}
