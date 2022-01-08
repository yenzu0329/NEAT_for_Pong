from pong import *
import pickle, gzip
import neat
import os
import visualize
os.environ["PATH"] += os.pathsep + 'C:\\Users\\user\\miniconda3\\envs\\pytorch\\Library\\bin\\graphviz'

UP = 0
DOWN = 1
STOP = 2
GEN = 0

def save_object(obj, filename):
    with gzip.open(filename, 'w', compresslevel=5) as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with gzip.open(filename) as f:
        obj = pickle.load(f)
        return obj

def fitness(genomes, config):
    global GEN
    nets = []
    ge = []
    game_managers = []
    GEN += 1

    for _, g in genomes:
        tmp_color = (randint(80,220),randint(80,220),randint(80,220))
        player = Player(WIDTH - 10, HEIGHT/2, tmp_color)
        opponent = Opponent(5, HEIGHT/2, tmp_color)
        paddles = [player, opponent]
        ball = Ball(randint(int(WIDTH*0.4),int(WIDTH*0.6)), randint(int(HEIGHT*0.4),int(HEIGHT*0.6)),
                     color = tmp_color, paddles = paddles)
        game_manager = GameManager(ball=ball, player=player, opponent=opponent)

        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        game_managers.append(game_manager)
        g.fitness = 0
        ge.append(g)
    
    run = True
    while run and len(game_managers) > 0:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                sys.exit()
        
        for i, gm in enumerate(game_managers):
            outputs = nets[i].activate((abs(gm.player.get_x()-gm.ball.get_x()), 
                                        gm.player.get_y(), 
                                        gm.ball.get_y(), 
                                        gm.ball.get_vel_direction()))
            action = outputs.index(max(outputs))
            if(action == UP):           gm.player.move_up()
            elif(action == DOWN):       gm.player.move_down()
            else:                       gm.player.stop()
        
        # Background Stuff
        screen.fill(bg_color)
        pygame.draw.rect(screen,light_grey,middle_strip)
        score_label = basic_font.render("Gens: "+str(GEN-1), True, light_grey)
        screen.blit(score_label, (10, 10))
        
        # Run the game
        for i, gm in enumerate(game_managers):
            reward = gm.run_game()
            ge[i].fitness += 0.01
            if ge[i].fitness > 500:
                run = False
                break
            if reward == -1:
                ge[i].fitness -= 0.01
                nets.pop(i)
                winner = ge.pop(i)
                game_managers.pop(i)

            elif reward == 1:
                ge[i].fitness += 3.0
    
        # Rendering
        pygame.display.flip()
        clock.tick(100)

    if len(game_managers) != 0:
        for g in ge:
            if g.fitness > winner.fitness:  winner = g
    node_names = {-1: 'distance_x', -2: 'player_y', -3: 'ball_y', -4: 'ball_direction',
            0: 'UP', 1: 'DOWN', 2:'STOP'}
    visualize.draw_net(config, winner, view=False, node_names=node_names, filename='net_'+str(GEN-1))

def run(config_path, model_name):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                    neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    winner = p.run(fitness_function=fitness, n=1000)
    print('\nBest genome:\n{!s}'.format(winner))
    print('save the network as {}'.format(model_name))
    save_object(winner, model_name)

    node_names = {-1: 'distance_x', -2: 'player_y', -3: 'ball_y', -4: 'ball_direction',
                  0: 'UP', 1: 'DOWN', 2:'STOP'}
    visualize.draw_net(config, winner, view=True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    model_name = os.path.join(local_dir, 'winner.pkl')
    run(config_path, model_name)
