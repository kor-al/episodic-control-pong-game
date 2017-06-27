import unreal_engine as ue
import time
import os
import logging
from trainer import VAETrainer
import numpy as np
from PIL import Image

path = "D:/Alice/Documents/HSE/masters"
# Experiment description.
GAME = "pong"
MODEL = "EC"
VERSION = 0  # Bump this for each new experiment.

EXPERIMENT_PATH = os.path.join(path,"/experiment", GAME, MODEL, str(VERSION))
MODEL_PATH = os.path.join(EXPERIMENT_PATH, "checkpoints")
LOG_PATH = os.path.join(EXPERIMENT_PATH, "logs")
SNAPSHOT_PERIOD = 10000  # periodicity of saving current model

config = {
	"batch_size": 100,  # size of minibatch
	"experiment_path": EXPERIMENT_PATH,
	"latent_dim": 32,
	"log_period": 100,
	"memory_size": 100000,
	"image_h": 84,
	"image_w": 84,
	"max_frames": 70000
}


class PythonVAEController(object):
# Called at the started of the game
	
	def __init__(self):
		self.trainer = VAETrainer(config)
		self.step_count = 0

	def begin_play(self):
		ue.log("Begin Play on PythonAIController class")
		self.trainer.init_training()
		self.trainer.load_model(MODEL_PATH)

		self.image_h, self.image_w = config["image_h"], config["image_w"]
		self.max_frames = config["max_frames"]
		self.screen_capturer = self.uobject.get_property('ScreenCapturer_Ref')

	def get_screen(self):
		screen = self.screen_capturer.MergedScreenshot
		if len(screen) == 0:
			return None
		return np.array(screen, dtype=np.uint8).reshape((self.image_h, self.image_w,1), order = 'F').swapaxes(0, 1)

	def set_state(self, state):
		self.screen_capturer.State = state.tolist()

	def tick(self, delta_time):
		#ue.log("Tick on Pycontroller")
		start_time = time.clock()

		screen = self.get_screen()

        # Skip frames when no screen is available.
		if screen is None:
			return
		
		#im = Image.fromarray(screen.reshape(self.image_h, self.image_w), 'L')
		#im.save("D:/your_file.png")

		if self.screen_capturer.bLearningMode and self.step_count < self.max_frames:
			
			state = self.trainer.process_frame(screen)

			self.step_count += 1

			if self.step_count % SNAPSHOT_PERIOD == 0:
				ue.log("Saving model...")
				self.trainer.save_model(MODEL_PATH)

            # Log elapsed time.
			finish_time = time.clock()
			elapsed = finish_time - start_time

		elif  self.step_count == self.max_frames:
			self.screen_capturer.bLearningMode = False

		state = self.trainer.get_state(screen)
		self.set_state(state)

	
