import random

target_sentence = "I love machine learning"

## Gene_pool establishment
gene_pool = " "
for i in range(65,122):
    gene_pool += chr(i)
print(gene_pool)

population_size = 10

#Generate Initial Population
def generate_chromosome(length):
    genes = []
    while len(genes) < length:
        genes.append(gene_pool[random.randrange(0,len(gene_pool))])
    return ''.join(genes)

#Random function
def randomfunction(end):
    return random.randrange(0,end)

#Calculate Fitness
def calculate_fitness(chromosome):
    fitness =0
    for i in range(len(chromosome)):
        if chromosome[i] == target_sentence[i]:
            fitness +=1
    return fitness

def calculate_population_fitness(population):
    population_fitness = []
    for chromosome in population:
        population_fitness.append(calculate_fitness(chromosome))
    return population_fitness

#Crossover function
def crossover(chromosome1, chromosome2):
    crossover_loc = random.randrange(0,len(chromosome1)-1)
    chromosome1_first = chromosome1[0:crossover_loc]
    chromosome1_second = chromosome1[crossover_loc:]
    chromosome2_first = chromosome2[0:crossover_loc]
    chromosome2_second = chromosome2[crossover_loc:]
    chromosome1_final = chromosome1_first + chromosome2_second
    chromosome2_final = chromosome2_first + chromosome1_second
    return chromosome1_final, chromosome2_final

def find_two_healthest_parents(population):
    population_modify = population.copy()
    population_modify_fitness = calculate_population_fitness(population_modify)
    healthest1th = max(population_modify_fitness)
    population_modify_fitness.remove(healthest1th)
    healthest2nd = max(population_modify_fitness)
    population_modify_fitness = calculate_population_fitness(population_modify)
    chromosome1 = population_modify[population_modify_fitness.index(healthest1th)]
    chromosome2 = population_modify[population_modify_fitness.index(healthest2nd)]
    chromosome1_index = population.index(chromosome1)
    chromosome2_index = population.index(chromosome2)
    return chromosome1_index, chromosome2_index, chromosome1, chromosome2

def find_two_unhealthest_chromoome(population):
    population_modify = population.copy()
    population_modify_fitness = calculate_population_fitness(population_modify)
    healthest1th = min(population_modify_fitness)
    population_modify_fitness.remove(healthest1th)
    healthest2nd = min(population_modify_fitness)
    population_modify_fitness = calculate_population_fitness(population_modify)
    chromosome1 = population_modify[population_modify_fitness.index(healthest1th)]
    chromosome2 = population_modify[population_modify_fitness.index(healthest2nd)]
    chromosome1_index = population.index(chromosome1)
    chromosome2_index = population.index(chromosome2)
    return chromosome1_index, chromosome2_index, chromosome1, chromosome2

def crossover_population(population):
    chromosome1_index, chromosome2_index, chromosome1, chromosome2 = find_two_healthest_parents(population)
    chromosome1_new, chromosome2_new = crossover(chromosome1, chromosome2)
    population[chromosome1_index] = chromosome1_new
    population[chromosome2_index] = chromosome2_new
    return population

#Mutate the new generation
def mutate_decide(probability):
    if random.random() < probability:
        return True
    else:
        return False

def mutate(chromosome):
    index_to_mutate = randomfunction(len(chromosome))
    gene = list(chromosome)
    mutated_gene = gene_pool[randomfunction(len(gene_pool))]
    gene[index_to_mutate] = mutated_gene
    return ''.join(gene)

def mutate_population(population, mutate_probability):
    chromosome1_index, chromosome2_index, chromosome1, chromosome2 = find_two_healthest_parents(population)
    chromosome3_index, chromosome4_index, chromosome3, chromosome4 = find_two_unhealthest_chromoome(population)
    if mutate_decide(mutate_probability):
        chromosome1_new = mutate(chromosome1)
        population[chromosome3_index] = chromosome1_new
    if mutate_decide(mutate_probability):
        chromosome2_new = mutate(chromosome2)
        population[chromosome4_index] = chromosome2_new
    return population

def evolution_criteria(population):
    for fittness in calculate_population_fitness(population):
        if fittness == len(target_sentence):
            return True
    return False
            
## TEST
# chromosome1 = 'eeeeee'
# chromosome2 = 'wwwwww'
# population = [chromosome1, chromosome2]
# crossover_population(population)
# print(population)

# Generation
def bug_discover(population):
    size = []
    for chromosome in population:
        size.append(len(chromosome))
    return size


def generation_algorithms(generation_times, mutate_probability):
    population = []
    for i in range(population_size):
        population.append(generate_chromosome(len(target_sentence)))
    population_fitness = calculate_population_fitness(population)   

    for generation in range(generation_times):

        # Crossover
        population = crossover_population(population)

        # Mutate
        population = mutate_population(population, mutate_probability)

        if evolution_criteria(population):
            break

        # print(bug_discover(population))
        print(generation)

    print("Current Population: ", population)
    print("Current Fitness", calculate_population_fitness(population))


generation_algorithms(generation_times = 100000, mutate_probability = 0.3)