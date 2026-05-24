type Coordinate = (number, number);

type Generic<T> = (number, T[])

type HasName = { name: string }
type HasAge = { age: number }

type Person = HasName & HasAge

class Player(name: string, health: number, speed: number)
    assert(health > 0, "Initial health must be greater than zero")

    function isDead(): boolean
        return self.health <= 0
    end

    function takeDamage(amount: number)
        self.health = self.health - amount
        if self:isDead() then
            println(self.name .. " has been defeated!")
        end
    end
end

let hero = Player("Arthur", 100.0, 16.5)

hero:takeDamage(45.0)

let healingFactor: number = 2.5 * 4.0 ^ 0.5 + 1.0
println(healingFactor)
