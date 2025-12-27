from Flatworld import *
from FuzzyFlatworld import *

class SubsumptionBot(FuzzyLogicBot):
    def __init__(self, position):
        super().__init__(position)

        self.modules = SubsumptionArchitecture(self)
        self.wanderBehavior = \
            self.modules.newBehavior(WanderBehavior)
        self.feedBehavior = \
            self.modules.newBehavior(FeedBehavior)
        self.fleeBehavior = \
            self.modules.newBehavior(FleeBehavior)

    def tick(self, secondsSinceLastFrame):
        self.spots = getSpots(self)
        self.isCloseIn = getCloseIn(self)
        self.calcLogic( )
        self.handleScared( )

        self.modules.tick( )
        super().tick(secondsSinceLastFrame)

    def moveDirection(self):

        direction = self.modules.moveDirection( )

        return direction

class SubsumptionArchitecture(object):
    def __init__(self, bot):
        self.behaviors = [ ]
        self.bot = bot

    def newBehavior(self, behaviorType):
        b = behaviorType(self.bot)
        self.behaviors.append(b)
        return b

    def enableAll(self):
        for behavior in self.behaviors:
            behavior.enable( )

    def moveDirection(self):
        for b in self.behaviors:
            r = b.move( )
            if r != None:
                return r

    def tick(self):
        self.enableAll( )
        for b in self.behaviors:
            b.tickFunction( )

class Behavior(object):
    def __init__(self,  bot):
        self.enabled = True
        self.bot = bot

    def enable(self):
        self.enabled = True
        
    def disable(self):
        self.enabled = False

    def move(self):
        if self.enabled:
            return self.moveFunction( )
        else:
            return None

    def tick(self):
        if self.enabled:
            self.tickFunction( )

    def moveFunction(self):
        pass
    
    def tickFunction(self):
        pass

class WanderBehavior(Behavior):
    def __init__(self,  bot):
        super().__init__(bot)
        self.lastDirection = pygame.Vector2(1,0)

    def moveFunction(self):
        if random.randint(0, 100) < 10:
            x = random.randint(-1, 1)
            y = random.randint(-1, 1)
            self.lastDirection = pygame.Vector2(x,y)
        return self.lastDirection

    def tickFunction(self):
        self.bot.color = 'blue'

class FeedBehavior(Behavior):
    def moveFunction(self):
        move = pygame.Vector2( )
        if self.bot.isGoodIn[N] > self.bot.isGoodIn[S]:
            move.y = -1
        else:            
            move.y = 1

        if self.bot.isGoodIn[W] > self.bot.isGoodIn[E]:
            move.x = -1
        else:          
            move.x = 1
        return move

    def tickFunction(self):
        goodInWorld = Fuzzy(0)
        for isGood in self.bot.isGoodIn:
            goodInWorld |= isGood

        if goodInWorld > Fuzzy(0.001) and \
            self.bot.speed < 500:

            self.bot.wanderBehavior.disable( )
            self.bot.color = 'cyan'

class FleeBehavior(Behavior):
    def moveFunction(self):
        D = ['W', 'NW', 'N', 'NE', 'E', 'SE', 'S', 'SW']

        posX = self.bot.position.x
        posY = self.bot.position.y

        maxX = getWidth( )
        maxY =getHeight( )
        
        half = Fuzzy(0.5)

        isCloseIn = [closeTo(posX),
                    closeTo(posY) | closeTo(posX),
                    closeTo(posY),
                    closeTo(maxX - posX) | closeTo(posY),
                    closeTo(maxX - posX),
                    closeTo(maxX - posX) | closeTo(maxY - posY),
                    closeTo(maxY - posY),
                    closeTo(posX) | closeTo(maxY - posY)]
                    
        isCloseIn = list(map(lambda x: x & half, isCloseIn))

        for i in range (8):

            a = i * 45 - 180
            low = a - 22.5
            high = a + 22.5

            before = (i + 7) % 8
            beforeBefore = (i + 6) % 8
            after = (i + 1) % 8
            afterAfter = (i + 2) % 8

            for spot in self.bot.spots:
                if spot.kind == 'danger':

                    bearing = spot.bearing

                    if bearing < -180:
                        bearing = bearing + 360

                    if bearing > 157.5:
                        bearing = bearing - 360

                    if bearing >= low and bearing <= high:
                        danger = closeTo(spot.distance)

                        isCloseIn[i] |=  danger
                        isCloseIn[before] |= danger & Fuzzy(0.75)
                        isCloseIn[after] |= danger & Fuzzy(0.75)
                        isCloseIn[beforeBefore] |= danger &  half
                        isCloseIn[afterAfter] |= danger & half

        safest = None
        danger = Fuzzy(1)
        for i in range(8):
            if isCloseIn[i] < danger:
                danger = isCloseIn[i]
                safest = i
        if safest == None:
                print('PANIC!')
                return pygame.Vector2( )

        direction = [ (-1,0), (-1,-1), (0,-1), (1, -1), \
                         (1, 0), (1,1), (0,1), (-1,1)]
        return pygame.Vector2(direction[safest])

    def tickFunction(self):
        if self.bot.scared > Fuzzy(0.2):
            self.bot.wanderBehavior.disable( )
            self.bot.feedBehavior.disable( )
            self.bot.color = 'orange'


if __name__ == '__main__':
    
    pygame.init()
    
    ticks = mainLoop(initialSpawns = {Red: 4,
                                       SubsumptionBot: 10,
                                       Grass : 60},
                      spawnInterval=600,
                      periodicSpawns={Grass: 60},
                      boardSize=(750,500),
                      playerType=Blue)
    print('That game lasted', ticks, 'ticks')

    pygame.quit()
