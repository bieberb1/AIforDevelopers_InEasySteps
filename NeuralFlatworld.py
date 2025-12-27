from operator import attrgetter
from functools import reduce

from Flatworld import *
from Insects import Insect, displayText

import tensorflow as tf
import numpy as np

NUM_DIRECTIONS = 2
NUM_VARIANTS = 3
NUM_EXTRAS = 6
NUM_OUTPUTS = 1
NUM_PERCEPTRONS = NUM_DIRECTIONS * NUM_VARIANTS \
                               + NUM_EXTRAS + NUM_OUTPUTS

NUM_MUTATIONS = 4

POPULATION = 40
CARNIVORE_FRACTION = 0.5

GENERATIONS = 10000

class Spot:
    def __init__(self, creature, thing):
        h = getHeight( )
        w = getWidth( )
        h2 = h / 2
        w2 = w / 2
        
        self.zones = [0.0] * NUM_DIRECTIONS

        vector = thing.position - creature.position

        if vector.x > w2:
            vector.x -= w

        if vector.y > h2:
            vector.y -= h

        self.distance, self.bearing = vector.as_polar( )

        _, sightAngle = creature.direction.as_polar( )
        bearing = self.bearing - sightAngle

        if bearing > -45 and bearing < 10:
            self.zones[0] = 10.0 / self.distance \
                                      if self.distance > 10 else 1.0

        elif bearing < 45 and bearing > -10:
            self.zones[1] = 10.0 / self.distance \
                                      if self.distance > 10 else 1.0

def getSpots(me):
    organisms = getOrganisms()
    redList = filter(lambda x: type(x) == Carnivore and \
                                        x != me,
                         organisms)

    blueList = filter(lambda x: type(x) == Herbivore and \
                                         x != me,
                          organisms)

    grassList = filter(lambda x: type(x) == Grass,
                           organisms)

    redSpots = list(map(lambda x: Spot(me, x),
                                    redList))

    blueSpots = list(map(lambda x: Spot(me, x),
                                     blueList))

    grassSpots = list(map(lambda x: Spot(me, x), grassList))

    return redSpots, blueSpots, grassSpots

def addTotals(a,b):
    return map(lambda a,b: a + b, a, b)

def getZones(spots):
    if len(spots) == 0:
        return [0.0] * NUM_DIRECTIONS
    zonesList = map(attrgetter('zones'), spots)
    zonesTotals = reduce(lambda a,b: addTotals(a,b),
                                  zonesList)
    return zonesTotals

def caption(x):
    if x < NUM_OUTPUTS:
        s = ['D'][x]
    elif x < NUM_OUTPUTS + NUM_EXTRAS:
        x = x - NUM_OUTPUTS
        s = 'E' + str(x)
    else:
        x = x - NUM_OUTPUTS - NUM_EXTRAS
        s = ['R', 'L'][x % NUM_DIRECTIONS]
        s = s + ['C', 'H', 'G'][x // NUM_DIRECTIONS]

    return s

class Chromosome:
    def __init__(self, chromosome=None):
        if chromosome == None:
            matrix = [ ]
            for y in range(NUM_PERCEPTRONS):
                matrix.append([random.random( )*2-1 for x in \
                                       range(NUM_PERCEPTRONS)])
            self.gene=tf.constant(matrix)
        else:
            self.gene = tf.constant(chromosome.gene)

    def mutate(self):
        mine = np.array(self.gene)
        for i in range(NUM_MUTATIONS):
            x = random.randint(0,NUM_PERCEPTRONS - 1)
            y = random.randint(0,NUM_PERCEPTRONS - 1)
            mine[y][x] *= (random.random( ) * 1.5 + 0.5) * \
                                  (random.randint(0, 1) * 2 - 1)
        self.gene = tf.constant(mine)
        return self

    def mate(self, other):
        child = Chromosome(self)
        mask = tf.random.uniform(self.gene.shape, 
                                               minval=0, maxval=2,
                                               dtype=tf.int32)
        mask = mask == 0
        child.gene = tf.where(mask, self.gene, other.gene)
        return child

    def __str__(self):
        s = '<'
        for op in range(NUM_PERCEPTRONS):
            s = s + caption(op) + ' = '
            for ip in range(NUM_PERCEPTRONS):
                s = s + caption(ip) + \
                      '{0:+8.3g}'.format(self.gene[ip][op]) + '   '
            s = s[:-1]+'\n '
        s = s[:-2] + '>'
        return s

def runNetwork(perceptrons, weights):
    states0 = tf.constant(perceptrons)
   
    inputs = tf.math.reduce_sum(states0 * weights, axis=1)
    states1 = tf.math.sigmoid(inputs)
            
    return states1

class Creature(Insect):
    def __init__(self, chromosome, position, color):
        super().__init__(None, position, 500, color)
        self.chromosome = chromosome
        self.score = 0

        self.extraPerceptrons = [random.gauss(0, 1) for i in \
                                                   range(NUM_EXTRAS)]

        self.direction = pygame.Vector2((-1, 0))
        self.direction = self.direction.rotate( \
                                             45 * random.randint(0, 9))

        self.redZones = [ ]
        self.blueZones = [ ]
        self.grassZones = [ ]

    def moveTo(self, newPosition):
        if newPosition.x < 0:
            newPosition.x += getWidth()

        if newPosition.x >= getWidth():
            newPosition.x = newPosition.x - getWidth() 

        if newPosition.y < 0:
            newPosition.y += getHeight()

        if newPosition.y >= getHeight():
            newPosition.y = newPosition.y - getHeight()

        organism = anyCollision(newPosition,
                                 ignore=self, ignoreType=type(self))

        if organism != None:
            return organism
        else:   
            self.position = newPosition
            return None

    def moveDirection(self):
        redSpots, blueSpots, grassSpots = getSpots(self)
        self.redZones = list(getZones(redSpots))
        self.blueZones = list(getZones(blueSpots))
        self.grassZones = list(getZones(grassSpots))

        self.perceptrons = tf.constant( \
              [0.0 for i in range(NUM_OUTPUTS)] +
              self.extraPerceptrons +
              self.redZones + self.blueZones + self.grassZones)

        result = runNetwork(self.perceptrons,
                                     self.chromosome.gene)

        outputs = np.array(result[:NUM_OUTPUTS])
        self.extraPerceptrons = \
                list(np.array(result[NUM_OUTPUTS : \
                                  NUM_OUTPUTS + NUM_EXTRAS]))

        self.rotation = (outputs[0] - 0.5) * 180
            
        return self.direction.rotate(self.rotation)

class Carnivore(Creature):
    def __init__(self, chromosome, position):
        super().__init__(chromosome, position,\
                              pygame.Color('red'))

    def hit(self, blocker):
        if type(blocker) == Herbivore:
            blocker.getBitten()
            self.score += 1

class Herbivore(Creature):
    def __init__(self, chromosome, position):
        super().__init__(chromosome, position, \
                              pygame.Color('blue'))

    def getBitten(self):
        self.score -= 1
        getOrganisms( ).remove(self)

    def hit(self, blocker):
        if type(blocker) == Grass:
            blocker.getBitten()
            self.score += 0.5

def pickOne(highest):
    x = abs(random.random( ) + random.random( ) - 1)
    return int(x * highest)

def nextGeneration(creatures, population):
    result = []
    thru = len(creatures) // 3
    for i in range(thru):
        result.append(creatures[i].chromosome)

    for i in range(thru, population):
        a = pickOne(len(creatures))
        b = pickOne(len(creatures))
        chromoA = creatures[a].chromosome
        chromoB = creatures[b].chromosome

        result.append(chromoA.mate(chromoB).mutate( ))

    return result

def countCreatures(creatures, kind):
    return len(list(
                   filter(lambda a: type(a) == kind, creatures)))

def oneGeneration(clock,
                           redChromosomes, blueChromosomes,
                           ticks, generation):

    global organisms, field, enableDraw

    field = getField( )
    organisms = [ ]
    setOrganisms(organisms)
    frameDecimation = 100
    redStartPopulation = len(redChromosomes)
    blueStartPopulation = len(blueChromosomes)

    for chromosome in redChromosomes:
        organisms.append(
                          Carnivore(chromosome, randomDrop( )))

    for chromosome in blueChromosomes:
        organisms.append(
                         Herbivore(chromosome, randomDrop( )))

    for i in range(60):
        organisms.append(Grass(randomDrop( )))

    allOrganisms = list(organisms)

    for tickCounter in range(ticks):
        if tickCounter % frameDecimation == 0:
            enableDraw = True

        redPop = countCreatures(organisms, Carnivore)
        bluePop = countCreatures(organisms, Herbivore)
        if bluePop < 1:
            break

        if enableDraw:
            clock.tick(60)
            field.fill('white')
        secondsSinceLastFrame  = 1/60.0
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None, tickCounter

        for organism in organisms:
            organism.tick(secondsSinceLastFrame)

        if enableDraw:
            displayText('Generation: ' + str(generation) +
                            ' Frame: ' + str(tickCounter) +
                            ' Pop: ' + str(redPop) +
                            ', ' + str(bluePop),
                            getField(), pygame.Vector2(0,0))
            pygame.display.flip()

        keys = pygame.key.get_pressed()
        if not keys[pygame.K_SPACE]: enableDraw = False

    print('population dropped to', redPop, bluePop)
    return allOrganisms, tickCounter

def neuralMainLoop(boardSize, population, generations):
    global field
    
    field = pygame.display.set_mode(boardSize)
    setField(field)
    
    numReds = int(population * CARNIVORE_FRACTION)
    numBlues = int(population * (1 - CARNIVORE_FRACTION))
    print ('numReds:', numReds, 'numBlues:', numBlues)
    
    redChromosomes = [Chromosome( ) \
                                              for i in range(numReds)]
    blueChromosomes = [Chromosome( ) \
                                             for i in range(numBlues)]

    clock = pygame.time.Clock()
    score = 0

    for generation in range(1, generations + 1):
       print('\nGeneration', generation)

       repeats = 200
       if generation > 20: repeats = 300
       if generation > 60: repeats = 500
       if generation > 100: repeats = 800
       if generation > 500: repeats = 500
       if generation > 1000: repeats = 300

       creatures, frameCount = \
           oneGeneration(clock,
                             redChromosomes, blueChromosomes, 
                             repeats, generation)
       if creatures == None:
           return score
       print('Duration:', frameCount, 'frames')

       reds = list(filter(
                                 lambda a: type(a) == Carnivore,
                                 creatures))
       blues = list(filter(
                                 lambda a: type(a) == Herbivore,
                                 creatures))

       if len(reds) > 0:
           reds = sorted(reds,
                             key=attrgetter('score'), reverse=True)

           redChromosomes = nextGeneration(reds, numReds)

           redScore = reduce(lambda a, b: a + b.score,
                                     reds, 0) / len(reds)
           print('Red chromosome:\n', str(reds[0].chromosome))
           print ('Red score average:', redScore)

       if len(blues) > 0:
           blues = sorted(blues,
                             key=attrgetter('score'), reverse=True)

           blueChromosomes =nextGeneration(blues,numBlues)

           blueScore = reduce(lambda a, b: a + b.score,
                                     blues, 0) / len(blues)
           print('Blue chromosome:\n',str(blues[0].chromosome))
           print ('Blue score average:', blueScore)



if __name__ == '__main__':
    
    pygame.init()
    
    neuralMainLoop(boardSize=(700,500), \
                           population=POPULATION, \
                          generations=GENERATIONS)

    pygame.quit()