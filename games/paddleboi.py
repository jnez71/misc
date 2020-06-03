#!/usr/bin/env python3
"""
PaddleBoi? PaddleBoi.

"""
import os; os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame  # pip3 install pygame
from numpy import array, clip, sqrt, sin, cos, arctan2, pi as PI
from numpy.random import uniform

##################################################

class PaddleBoi:
    """
    |P A D D L E B O I|
    Move your paddleboi with the arrow keys (or "WASD").
    A wild ball bounces around. Help it hit green zones to improve your life.

    """
    #### GLOBALS
    # Hashmap from valid decisions to corresponding action values
    CHOICES = {"stall": ( 0,  0),
                "left": (-1,  0),
               "right": ( 1,  0),
                  "up": ( 0, -1),
                "down": ( 0,  1)}
    ####

    def __init__(self, width=500, height=500,
                 boi_size=20, boi_speed=5,
                 ball_size=20, ball_speed=5,
                 zone_size=20):
        """
        PaddleBoi constructor.

        width: extent of the domain's x-dimension (pix)
        height: extent of the domain's y-dimension (pix)
        boi_size: radius of paddleboi's circular hitbox (pix)
        boi_speed: max distance covered by paddleboi per update (pix/step)
        ball_size: radius of the ball's circular hitbox (pix)
        ball_speed: max distance covered by the ball per update (pix/step)
        zone_size: radius of the zone's circular hitbox (pix)

        """
        # Store provided parameters
        self.width = int(width)
        self.height = int(height)
        self.boi_size = int(boi_size)
        self.boi_speed = int(boi_speed)
        self.ball_size = int(ball_size)
        self.ball_speed = int(ball_speed)
        self.zone_size = int(zone_size)
        # Establish initial game data
        self.reset()

    def reset(self):
        """
        Sets game data to default initial conditions.

        """
        # Set paddle position to center of domain
        self.boi_x = (self.width - self.boi_size) // 2
        self.boi_y = self.height // 2
        # Set random but valid ball position and direction
        self.ball_x = int(uniform(self.ball_size, self.width - self.ball_size))
        self.ball_y = int(uniform(self.ball_size, self.height - self.ball_size))
        self.ball_dx = cos(uniform(0.0, 2*PI))
        self.ball_dy = sqrt(1.0 - self.ball_dx**2)
        # Set random but valid zone position
        self.zone_x = int(uniform(self.zone_size, self.width - self.zone_size))
        self.zone_y = int(uniform(self.zone_size, self.height - self.zone_size))
        # Set life-improvement score to minimum
        self.score = 0

    def get_state(self):
        """
        Returns an array of floats that encode your current situation.

        """
        return array((self.boi_x,
                      self.boi_y,
                      self.ball_x,
                      self.ball_y,
                      arctan2(self.ball_dy, self.ball_dx),
                      self.zone_x,
                      self.zone_y), dtype=float)

    def update(self, decision):
        """
        Advances the game forward by one step using the provided decision.
        Returns the reward earned for this transition.

        decision: any valid member of CHOICES, it's up to you

        """
        # Turn symbolic decision into an action value, if valid
        try:
            action = PaddleBoi.CHOICES[decision]
        except KeyError:
            raise KeyError("Invalid decision provided to update PaddleBoi. "
                           +"Choose from: {0}.".format(PaddleBoi.CHOICES.keys()))
        # Update paddle position based on speed parameter and decided action
        self.boi_x += action[0]*self.boi_speed
        self.boi_y += action[1]*self.boi_speed
        # Prevent paddle from leaving the domain
        self.boi_x = clip(self.boi_x, self.boi_size, self.width - self.boi_size)
        self.boi_y = clip(self.boi_y, self.boi_size, self.height - self.boi_size)
        # Update ball position
        self.ball_x = int(clip(self.ball_x + self.ball_speed*self.ball_dx, self.ball_size, self.width - self.ball_size))
        self.ball_y = int(clip(self.ball_y + self.ball_speed*self.ball_dy, self.ball_size, self.height - self.ball_size))
        # Reflect ball if hit boundaries
        if (self.ball_x <= self.ball_size) or (self.ball_x >= self.width - self.ball_size):
            self.ball_dx *= -1.0
        if (self.ball_y <= self.ball_size) or (self.ball_y >= self.height - self.ball_size):
            self.ball_dy *= -1.0
        # Reflect ball if hit paddleboi
        dist_x = self.boi_x - self.ball_x
        dist_y = self.boi_y - self.ball_y
        dist = sqrt(dist_x**2 + dist_y**2)
        if (dist < self.boi_size + self.ball_size) and (dist > 1e-6):
            # This is "meant" to be "broken"
            dir_x = dist_x / dist
            dir_y = dist_y / dist
            s = 2*(dir_x*self.ball_dx + dir_y*self.ball_dy)
            self.ball_dx -= s*dir_x
            self.ball_dy -= s*dir_y
            # Rebound
            self.boi_x -= 2*action[0]*self.boi_speed
            self.boi_y -= 2*action[1]*self.boi_speed
        # Renormalize
        norm = sqrt(self.ball_dx**2 + self.ball_dy**2)
        self.ball_dx /= norm
        self.ball_dy /= norm
        # Impose consequences
        dist_x = self.zone_x - self.ball_x
        dist_y = self.zone_y - self.ball_y
        dist = sqrt(dist_x**2 + dist_y**2)
        if dist < self.zone_size + self.ball_size:
            reward = 1
            self.zone_x = int(uniform(self.zone_size, self.width - self.zone_size))
            self.zone_y = int(uniform(self.zone_size, self.height - self.zone_size))
        else:
            reward = 0
        self.score += reward
        return float(reward)

    def display(self, screen):
        """
        Draws the current situation on the provided screen.

        screen: a Pygame.Surface you are okay with completely drawing over

        """
        # Set screen title to game name and current score
        pygame.display.set_caption("PADDLE BOI - {0}".format(self.score))
        # Wipe all pixels to black
        screen.fill((0, 0, 0))
        # Select a color representative of current score
        color = (int(255*(2.0**(-self.score/30))), int(255*(1.0 - 2.0**(-self.score/30))), 0)
        # Draw the paddle
        pygame.draw.circle(screen, color[::-1], (self.boi_x, self.boi_y), self.boi_size, 0)
        # Draw the ball
        pygame.draw.circle(screen, color, (self.ball_x, self.ball_y), self.ball_size, 0)
        # Draw the consequence zone
        pygame.draw.circle(screen, (0, 255, 0), (self.zone_x, self.zone_y), self.zone_size, 0)
        # Render the image
        pygame.display.update()

    def play(self, policy=None, rate=60):
        """
        Runs the game using the given policy function and at the specified update rate.
        If a policy is not given, then decisions are read live from human input.

        policy: function that takes a state vector and returns a valid decision
        rate: max number of updates (and display frames) per second of real time

        """
        # Create visualization of domain
        screen = pygame.display.set_mode((self.width, self.height))
        # Enter main loop
        playing = True
        while playing:
            # Record current real time (in milliseconds)
            start_time = pygame.time.get_ticks()
            # Draw current situation to the screen
            self.display(screen)
            # Use policy or human input to make decision
            if policy:
                decision = policy(self.get_state())
            else:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                    decision = "left"
                elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                    decision = "right"
                elif keys[pygame.K_UP] or keys[pygame.K_w]:
                    decision = "up"
                elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                    decision = "down"
                else:
                    decision = "stall"
            # Evolve the situation by one step with the chosen decision
            self.update(decision)
            # Check for quit or reset command events
            for event in pygame.event.get():
                if (event.type == pygame.QUIT) or\
                   ((event.type == pygame.KEYDOWN) and (event.key == pygame.K_q)):
                    playing = False
                    break
                elif (event.type == pygame.KEYDOWN) and (event.key == pygame.K_r):
                    self.reset()
            # Rest for amount of real time alloted per loop less what was used
            remaining_time = (1000//rate) - (pygame.time.get_ticks() - start_time)
            if remaining_time > 0:
                pygame.time.wait(remaining_time)

##################################################

if __name__ == "__main__":

    game = PaddleBoi()
    game.play()

##################################################
