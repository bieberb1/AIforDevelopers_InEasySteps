import random
import pygame
from math import *

def mainLoop(initialSpawns, spawnInterval, 
                     periodicSpawns, boardSize, playerType):
    global organisms, field
    
    field = pygame.display.set_mode(boardSize)
    clock = pygame.time.Clock( )
    running = True
    organisms = [ ]
    doSpawns(initialSpawns)
    tickCounter = 0

    while any([isinstance(organism, playerType) \
                       for organism in organisms]) \
            and running:
        tickCounter += 1
        secondsSinceLastFrame  = clock.tick(60) / 1000
        
        for event in pygame.event.get( ):
            if event.type == pygame.QUIT:
                running = False

        field.fill('white')

        for organism in organisms:
            organism.tick(secondsSinceLastFrame)

        if tickCounter % spawnInterval == 0:
            doSpawns(periodicSpawns)

        pygame.display.flip( )
    
    return tickCounter

def doSpawns(spawns):
    global organisms

    for sort in spawns:
        for i in range(spawns[sort]):
            organisms.append(sort(randomDrop( )))

def randomDrop():
    w = field.get_width( )
    h = field.get_height( )
    while True:
        position = pygame.Vector2(random.random( ) * w,
                                            random.random( ) * h)
        organism = anyCollision(position)
        if organism == None:
            return position

def anyCollision(position, ignore = None ,
                        ignoreType = type(None)):
    for organism in organisms:
        if organism != ignore and \
                    not isinstance(organism, ignoreType):
            if (organism.position - position).length() < 20:
                return organism
    return None

class Organism:
    def __init__(self, position):
        self.position = position

    def tick(self, secondsSinceLastFrame):
        self.draw(self.position)

    def getBitten(self):
        pass

    def bearing(self, other):
        return (other.position - self.position).as_polar( )

    def moveTo(self, newPosition):
        if newPosition.x < 0:
            newPosition.x = 0

        if newPosition.x >= field.get_width( ):
            newPosition.x = field.get_width( ) - 1

        if newPosition.y < 0:
            newPosition.y = 0

        if newPosition.y >= field.get_height( ):
            newPosition.y = field.get_height( ) - 1

        organism = anyCollision(newPosition, ignore=self)

        if organism != None:
            return organism
        else:   
            self.position = newPosition
            return None
            
class Creature(Organism):
    def __init__(self, position, speed, color):
        super( ).__init__(position)
        self.speed = speed
        self.color = color
    
    def draw(self, position):
        pygame.draw.circle(field, self.color, position, 10)

    def tick(self, secSinceLast):
        vector = self.moveDirection( )
        r, angle = vector.as_polar( )
        if r != 0:
            vector.scale_to_length(self.speed * secSinceLast)

        newPosition = self.position + vector
        blocker = self.moveTo(newPosition)
        if blocker != None:
            self.hit(blocker)
        super( ).tick(secSinceLast)

    def hit(self, blocker):
        pass
    
    def moveDirection(self):
        return pygame.Vector2( )

class Grass(Organism):
    def __init__(self, position):
        super( ).__init__(position)
        self.leaves = 5

    def getBitten(self):
        if self.leaves > 0:
            self.leaves -= 1
            if self.leaves <= 0:
                organisms.remove(self)

    def draw(self, position):
        a = int(position.x - 10)
        b = int(position.x + 10)
        c = int(position.x)

        top = int(position.y - 10)
        bottom = int(position.y + 10)

        s = 20 // self.leaves

        for x in range(a + s // 2, b, s):
            pygame.draw.line(field, 'green',
                                        (c, bottom), (x, top))
            
class Blue(Creature):
    def __init__(self, position):
        super().__init__(position, 300, 'blue')

    def getBitten(self):
        global creatures
        organisms.remove(self)

    def tick(self, secondsSinceLastFrame):
        self.speed *= 0.9999
        super().tick(secondsSinceLastFrame)

    def hit(self, blocker):
        if isinstance(blocker, Grass):
            self.speed += 1
            blocker.getBitten( )

class Player(Blue):
    def moveDirection(self):
        keys = pygame.key.get_pressed( )
        vector = pygame.Vector2( )
        if keys[pygame.K_w]:
            vector.y = -1
        if keys[pygame.K_s]:
            vector.y = 1
        if keys[pygame.K_a]:
            vector.x = -1
        if keys[pygame.K_d]:
            vector.x = 1

        return vector

class Red(Creature):
    def __init__(self, position):
        super().__init__(position, 200, 'red')

    def moveDirection(self):
        nearestDist = 2000.0
        nearestCreature = None
        for organism in organisms:
            if isinstance(organism, Blue):
                r,angle = self.bearing(organism)
                if r < nearestDist:
                    nearestDist = r
                    nearestCreature = organism

        if nearestCreature:
            return nearestCreature.position - self.position
        else:
            return pygame.Vector2( )

    def hit(self, blocker):
        if isinstance(blocker, Blue):
            blocker.getBitten()

def getHeight( ):     return field.get_height( )
def getWidth( ):     return field.get_width( )
def getOrganisms( ):     return organisms
def getField( ):     return field
def setOrganisms(o):
     global organisms
     organisms = o
def setField(f):
     global field
     field = f