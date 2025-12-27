from Flatworld import *

N = 0
S = 1
E = 2
W = 3

class Fuzzy:
    def __init__(self, value=0):
        self.value = max(0, min(1, value))

    def __and__(self, other):
        return Fuzzy(self.value * other.value)

    def __invert__(self):
        return Fuzzy(1 - self.value)

    def __or__(self, other):
        return ~((~self) & (~other))

    def __gt__(self, other):
        return self.value > other.value
        
    def __lt__(self, other):
        return self.value < other.value

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)

class Spot:
    def __init__(self, position, thing):
        vector = thing.position - position
        self.distance, self.bearing = vector.as_polar()

        self.direction = set()

        if self.bearing < 0:
            self.direction.add(N)

        if self.bearing > 0:
            self.direction.add(S)

        if abs(self.bearing) < 90:
            self.direction.add(E)

        if abs(self.bearing) > 90:
            self.direction.add(W)

        if isinstance(thing,Grass):
            self.kind = 'food'
            self.isClose = closeTo(self.distance)

        elif isinstance(thing, Red):
            self.kind = 'danger'
            self.isClose = closeTo(self.distance)

        elif isinstance(thing,Blue):
            self.kind = 'friend'
            self.isClose = closeTo(self.distance)

        else:
            print('Unknown thing:', thing)

def getSpots(me):
    spots = [ ]
    for organism in getOrganisms( ):
        if organism != me:
            s = Spot(me.position, organism)
            spots.append(s)
    return spots

def getCloseIn(me):
    isCloseIn = [ {'food': Fuzzy(0),
                   'danger': Fuzzy(0),
                   'friend' : Fuzzy(0)} for i in 'NSEW' ]

    for spot in me.spots:
        for d in range(4):
            if d in spot.direction:
                isCloseIn[d][spot.kind] |= spot.isClose
    return isCloseIn

class FuzzyLogicBot(Blue):
    def __init__(self, position):
        super().__init__(position)
        self.scared = Fuzzy(0)

    def calcLogic(self):
        self.isGoodIn = [ ]
        self.isBadIn = [ ]
        for isClose in self.isCloseIn:

            isGood = isClose['food'] & \
                        ~isClose['friend'] & \
                        ~isClose['danger']
            isBad = isClose['danger']

            self.isGoodIn.append(isGood)
            self.isBadIn.append(isBad)

    def handleScared(self):
        if self.isBadIn[N] |    \
                self. isBadIn[S] |  \
                self.isBadIn[E] |   \
                self.isBadIn[W] > \
                Fuzzy(0.5):

            self.scared = Fuzzy(1)
        else:
            self.scared &= Fuzzy(0.9)

    def moveDirection(self):
        self.spots = getSpots(self)
        self.isCloseIn = getCloseIn(self)
        self.calcLogic( )
        self.handleScared( )

        if self.scared > Fuzzy(0.3):
            return self.chooseScaredDirection( )
        else:
            return self.chooseUnscaredDirection( )

    def chooseScaredDirection(self):
        move = pygame.Vector2()

        if self.isBadIn[S] > self.isBadIn[N]:
            move.y = -1
        else:
            move.y = 1

        if self.isBadIn[E] > self.isBadIn[W]:
            move.x = -1
        else:
            move.x = 1
        return move

    def chooseUnscaredDirection(self):
        move = pygame.Vector2()

        if self.isGoodIn[N] > self.isGoodIn[S]:
            move.y = -1
        else:            
            move.y = 1

        if self.isGoodIn[W] > self.isGoodIn[E]:
            move.x = -1
        else:
             move.x = 1
        return move



def closeTo(dist):
    return Fuzzy(1 / (max(1,dist)/20)**2)

if __name__ == '__main__':
    pygame.init()

    ticks = mainLoop(initialSpawns = {Red: 4,
                                     FuzzyLogicBot: 10,
                                     Grass : 60},
                         spawnInterval=30,
                         periodicSpawns={Grass: 4},
                         boardSize=(750,500),
                         playerType=Blue)
    print('That game lasted', ticks,'ticks')   
    pygame.quit()