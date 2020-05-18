import pygame
import predict

pygame.init()

size = width, height =  500, 500
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

get_num_screen = pygame.display.set_mode(size)
pygame.display.set_caption("Pencil")
get_num_screen.fill(BLACK)
pygame.display.flip()

file = "drawing.jpeg"

clock = pygame.time.Clock()


def draw_circle(screen, colour, pos_x, pos_y, radius):
	pygame.draw.circle(screen, colour, (pos_x, pos_y), radius)	


def get_number():
	"""
	Provide the user with an interface to draw their number
	When the users is holding down mouse 1 draw a circle to the scren.
	Scale the surface down to 28 x 28 
	Save surface as "drawing.jpeg"
	"""
	global get_num_screen
	running = True

	while running:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				get_num_screen = pygame.transform.scale(get_num_screen, (28,28))
				pygame.image.save(get_num_screen, file)
				running = False

		# cursor position
		pos_x, pos_y = pygame.mouse.get_pos()
		# check if mouse button 1 is pressed
		if pygame.mouse.get_pressed() == (1,0,0):
			draw_circle(get_num_screen, WHITE, pos_x, pos_y, 10)
		
		clock.tick(6000)
		pygame.display.update()

def show_predict(guess):
	"""
	Display what the user drew scaled down to 28 x 28 pixels
	Display the networks predictions as well
	"""
	global predict_screen
	running = True
	pygame.display.set_caption("My Guess")
	predict_screen  = pygame.display.set_mode((250, 250))
	predict_screen.fill(WHITE)

	# load and rescale image to 250 by 250
	digit = pygame.image.load(file)	
	digit = pygame.transform.scale(digit, (250,200))

	font = pygame.font.Font('freesansbold.ttf', 40) 
	text = font.render('My Guess: {g}'.format(g = guess), True, GREEN, WHITE)

	# draw drawn digit and network guess to the screen
	predict_screen.blit(digit, (0,0))
	predict_screen.blit(text, (0, 200))

	pygame.display.update()

	while running:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False

		clock.tick(6000)
		pygame.display.update()



def main():
	get_number()
	guess = predict.run_predict()
	show_predict(guess)
	

if __name__ == "__main__":
    main()

