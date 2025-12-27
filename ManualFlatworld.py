from Flatworld import *

# ------ Main Program ------
pygame.init( )
ticks = mainLoop(initialSpawns = {Red: 4,
                                                Player: 1,
                                                Grass : 20},
                    spawnInterval=30,
                    periodicSpawns={Grass: 1},
                    boardSize=(750,500), playerType=Blue)

print('That game lasted', ticks, 'ticks')
pygame.quit( )
