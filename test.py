from main import *

def run_for_test(net):
    # Game objects
    tmp_color = (randint(80,220),randint(80,220),randint(80,220))
    player = Player(WIDTH - 10, HEIGHT/2, tmp_color)
    opponent = Opponent(5, HEIGHT/2, tmp_color)
    paddles = [player, opponent]
    ball = Ball(WIDTH/2, HEIGHT/2, color = tmp_color, paddles = paddles)
    game_manager = GameManager(ball=ball, player=player, opponent=opponent)
    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        outputs = net.activate((abs(player.get_x()-ball.get_x()), 
                                    player.get_y(), 
                                    ball.get_y(), 
                                    ball.get_vel_direction()))
        action = outputs.index(max(outputs))
        if(action == UP):           player.move_up()
        elif(action == DOWN):       player.move_down()
        else:                       player.stop()
        
        # Background Stuff
        screen.fill(bg_color)
        pygame.draw.rect(screen,light_grey,middle_strip)
        
        # Run the game
        game_manager.run_game()
        done = game_manager.is_done()

        # Rendering
        pygame.display.flip()
        clock.tick(60)

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    model_name = os.path.join(local_dir, 'winner.pkl')
    winner = load_object(model_name)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)

    winner_net = neat.nn.FeedForwardNetwork.create(winner, config) 
    run_for_test(winner_net)