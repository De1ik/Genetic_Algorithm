import matplotlib.pyplot as plt
import random

CREATE_PLOT = False
SELECTION_MODE = 1  # 1 -> Roulette / 0 -> Tournament


# example of the input map
# game_map = {
#     6: [0,0,0,0,0,0,0],
#     5: [0,0,0,0,1,0,0],
#     4: [0,0,1,0,0,0,0],
#     3: [0,0,0,0,0,0,1],
#     2: [0,1,0,0,0,0,0],
#     1: [0,0,0,0,1,0,0],
#     0: [0,0,0,"S",0,0,0]
#        }

# read input map and indentify start and gold points
def read_map_file():
    with open("map.txt", "r") as f:
        game_map_f = [line.strip() for line in f.readlines()]
        game_map = dict()
        index = len(game_map_f) - 1
        start = {"row": None, "col": None}
        treasures = []

        for i in range(len(game_map_f)):
            game_map[index] = game_map_f[i]
            if "S" in game_map_f[i] or "1" in game_map_f[i]:
                for j in range(len(game_map_f[i])):
                    if game_map_f[i][j] == "S":
                        start["row"] = index
                        start["col"] = j
                    if game_map_f[i][j] == "1":
                        treasures.append([index, j])
            index -= 1

        return game_map, start, treasures


# create start population
def init_population():
    addresses_vm = []
    for i in range(64):
        addresses_vm.append(create_random_address())
    return addresses_vm


# create random instruction
def create_random_address():
    type = random.randint(0, 3)
    if type == 0:
        number = random.randint(0, 63)
    elif type == 1:
        number = random.randint(64, 127)
    elif type == 2:
        number = random.randint(128, 191)
    else:
        number = random.randint(192, 254)
    bin_number = (bin(number)[2:]).zfill(8)
    return bin_number


# verify the movement
def vm_write(eight_bits: str, current_position: list, result: list):
    amount_of_1 = eight_bits.count("1")
    if amount_of_1 >= 7:
        result.append("D")
        current_position["row"] -= 1
    elif amount_of_1 >= 5:
        result.append("P")
        current_position["col"] += 1
    elif amount_of_1 >= 3:
        result.append("L")
        current_position["col"] -= 1
    else:
        result.append("H")
        current_position["row"] += 1


# check that the position inside the map
def check_position(current_position: list, treasures: list):
    if (current_position["row"] < 0 or current_position["row"] > 6 or current_position["col"] < 0 or current_position[
        "col"] > 6):
        return -1
    elif [current_position["row"], current_position["col"]] in treasures:
        return 1
    return 0


# evaluation of genes
def fitness(address_list, start, treasure):
    instr_link = 0
    instr_counter = 0
    current_position = start
    treasure_amount = len(treasure)
    default_treasure_amount = treasure_amount
    fitness_result = 1
    result = []
    while instr_counter < 500 and instr_link < len(address_list):
        address = address_list[instr_link]
        type = address[0:2]
        six_bits = address[2:]
        link_to_addr = int(six_bits, 2)
        if type == "00":
            new_value = (int(address_list[link_to_addr], 2) + 1) % 256
            address_list[link_to_addr] = (bin(new_value)[2:]).zfill(8)
        elif type == "01":
            new_value = (int(address_list[link_to_addr], 2) - 1) % 256
            address_list[link_to_addr] = (bin(new_value)[2:]).zfill(8)
            pass
        elif type == "10":
            instr_link = link_to_addr - 1
        else:
            vm_write(address_list[link_to_addr], current_position, result)
            result_move = check_position(current_position, treasure)

            if result_move == -1:
                instr_counter += 1
                break
            elif result_move == 1:
                treasure.remove([current_position["row"], current_position["col"]])
                treasure_amount -= 1
                fitness_result += 1000
                if treasure_amount == 0:
                    instr_counter += 1
                    break
        instr_link += 1
        instr_counter += 1

    return [fitness_result, default_treasure_amount - treasure_amount, instr_counter, result]


# roulette selection method
def roulette_select(population, fitnessed_population):
    total_fitness = sum(x[0] for x in fitnessed_population)
    pick = random.uniform(0, total_fitness)
    current = 0
    for i, fit in enumerate(fitnessed_population):
        current += fit[0]
        if current > pick:
            return population[i]


# tournament selection method
def tournament_select(population, fitnessed_population, k=10):
    point = random.randint(0, len(population[0]) - k - 1)
    zipped_array = list(zip(population, fitnessed_population))
    point_arr = zipped_array[point:point + k]
    selected = sorted(point_arr, key=lambda x: x[1][0], reverse=True)
    return selected[0][0]


# select individual by the selected method
def select(population, fitnessed_population):
    if SELECTION_MODE:
        return roulette_select(population, fitnessed_population)
    return tournament_select(population, fitnessed_population)


# crossover parents by a chance
def crossover(parent1, parent2, crossover_rate):
    if random.random() < crossover_rate:
        point = random.randint(1, 64 - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    return parent1, parent2


# mutate genes by a chance
def mutate(genome, mutation_rate):
    for i in range(len(genome)):
        if random.random() < mutation_rate:
            genome[i] = create_random_address()
    return genome


# main flow of the genetic algorithm
def main():
    game_map, start, treasures = read_map_file()
    treasure_size = 5
    generation_amount = 1000
    population_amount = 100
    mutation_rate = 0.1
    crossover_rate = 0.95
    elite_size = 5

    number_of_find_5 = 0
    number_of_tests = 100

    found_treasure = {
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
    }

    data_for_graf = [[0, 0, 0] for _ in range(generation_amount)]
    for j in range(1, number_of_tests + 1):
        found = False
        population = [init_population() for _ in range(population_amount)]

        for i in range(generation_amount):
            fitnessed_population = [fitness(addresses_vm.copy(), start.copy(), treasures.copy()) for addresses_vm in
                                    population]
            best_genome = max(fitnessed_population, key=lambda x: x[0])

            data_for_graf[i][2] += 1
            data_for_graf[i][1] += best_genome[1]
            data_for_graf[i][0] += best_genome[0]

            found_treasure[str(best_genome[1])] += 1

            if not found:
                if best_genome[1] == treasure_size:
                    number_of_find_5 += 1
                    print(f"Wave {j}: Generation {i}".ljust(
                        25) + f"|    The amount of treasure found = {best_genome[1]}".ljust(
                        38) + f"|    The number of completed instructions = {best_genome[2]}".ljust(
                        48) + f"|    The number of steps = {len(best_genome[3])}".ljust(
                        32) + f"|    Move steps: {' '.join(best_genome[3])}")
                    found = True
                    if not CREATE_PLOT:
                        break
            new_population = []
            for el in range(elite_size):
                new_population.append(population[fitnessed_population.index(
                    (sorted(fitnessed_population, key=lambda x: x[0], reverse=True))[el])])

            while len(new_population) < population_amount:
                parent1 = select(population, fitnessed_population)
                parent2 = select(population, fitnessed_population)
                child1, child2 = crossover(parent1, parent2, crossover_rate)

                new_population.append(mutate(child1, mutation_rate))
                if len(new_population) < population_amount:
                    new_population.append(mutate(child2, mutation_rate))

            population = new_population
            if i == generation_amount - 1 and best_genome[1] != treasure_size and not found:
                print(f"Wave {j}: Generation {i}".ljust(
                    25) + f"|    The amount of treasure found = {best_genome[1]}".ljust(
                    38) + f"|    The number of completed instructions = {best_genome[2]}".ljust(
                    48) + f"|    The number of steps = {len(best_genome[3])}".ljust(
                    32) + f"|    Move steps: {' '.join(best_genome[3])}")
    print(f"Percent of Success: {(number_of_find_5 / number_of_tests) * 100}% ")

    if CREATE_PLOT:
        # first plot with Generation / Average number of treasures
        avg_found_treasure_over_generations = [data[1] / data[2] if data[2] != 0 else 1 for data in data_for_graf]
        plt.figure(figsize=(10, 5))

        generations = list(range(1, len(avg_found_treasure_over_generations) + 1))
        plt.plot(avg_found_treasure_over_generations, generations, label="Average number of treasures found",
                 marker='o')

        plt.title("Evolution number of found treasures by generation")
        plt.xlabel("Amount of Found Treasures")
        plt.ylabel("Generations")
        plt.legend()
        plt.grid(True)

        plt.show()

        # second plot with size of treasure through all generations
        x_values = list(map(int, found_treasure.keys()))
        y_values = list(found_treasure.values())

        plt.figure(figsize=(8, 5))
        plt.bar(x_values, y_values)

        plt.xlabel("Treasures")
        plt.ylabel("Amount")
        plt.title("Found Treasures and their Amounts")

        for i in range(len(x_values)):
            plt.text(x_values[i], y_values[i], str(y_values[i]), ha='center', va='bottom')

        plt.show()


if "__main__" == __name__:
    main()
