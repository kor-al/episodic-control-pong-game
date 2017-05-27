import unreal_engine as ue
import time
import os
import logging
from trainer import VAETrainer
import numpy as np

from unreal_engine.classes import ActorComponent, ForceFeedbackEffect, KismetSystemLibrary


# Experiment description.
GAME = "pong"
MODEL = "EC"
VERSION = 0  # Bump this for each new experiment.

EXPERIMENT_PATH = os.path.join("/experiment/acid", GAME, MODEL, str(VERSION))
MODEL_PATH = os.path.join(EXPERIMENT_PATH, "checkpoints")
LOG_PATH = os.path.join(EXPERIMENT_PATH, "logs")
SNAPSHOT_PERIOD = 10000  # periodicity of saving current model

config = {
	"batch_size": 100,  # size of minibatch
	"experiment_path": EXPERIMENT_PATH,
	"latent_dim": 32,
	"log_period": 1000,
	"memory_size": 100000,
	"image_h": 84,
	"image_w": 84,
	"max_frames": 50000
}


class PythonVAEController(object):
# Called at the started of the game
	
	def __init__(self):
		self.trainer = VAETrainer(config)
	
	def begin_play(self):
		ue.log("Begin Play on PythonAIController class")
		self.step_count = 0
		self.trainer.init_training()
		self.trainer.load_model(MODEL_PATH)
		self.image_h, self.image_w = config["image_h"], config["image_w"]
		self.max_frames = config["max_frames"]

	def get_screen(self):
		screen_capturer = self.uobject.get_property('ScreenCapturer_Ref')
		screen = screen_capturer.MergedScreenshot
		if len(screen) == 0:
			return None
		return np.array(screen).reshape((self.image_h, self.image_w, 1), order='F')

	def tick(self, delta_time):
		ue.log("Tick on Pycontroller")
		start_time = time.clock()

		screen = self.get_screen()

        # Skip frames when no screen is available.
		if screen is None or len(screen) == 0:
			ue.log("zero screen")
			return

		if self.step_count < self.max_frames:
			
			self.trainer.process_frame(screen)

			self.step_count += 1

			if self.step_count % SNAPSHOT_PERIOD == 0:
				self.trainer.save_model(MODEL_PATH)

            # Log elapsed time.
			finish_time = time.clock()
			elapsed = finish_time - start_time

		else:
			state = self.trainer.get_state(screen)

        # Log elapsed time.
		finish_time = time.clock()
		elapsed = finish_time - start_time

		#return state
