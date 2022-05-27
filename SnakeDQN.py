# -------------------------
# Project: DQN and PRSO on Snake
# Author: yc_zhao
# Date: 2022.5.5
# -------------------------

import cv2
import sys
sys.path.append("game/")
import GluttonousSnake as game
from BrainDQN_Nature import BrainDQN
from BrainDQN_Nature1 import BrainDQN1
import numpy as np


# preprocess raw image to 80*80 gray image
def preprocess(observation):
	observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)#灰度转化
	ret, observation = cv2.threshold(observation,127,255,cv2.THRESH_BINARY)
	# cv2.imwrite("3.jpg",observation,[int(cv2.IMWRITE_JPEG_QUALITY),5])#输出二值化的图片
	return np.reshape(observation,(80,80,1))

def playSnake():
	# Step 1: init BrainDQN
	actions = 4
	actions1 = 4
	top = 0
	top1 = 0
	brain = BrainDQN(actions)
	brain1 = BrainDQN1(actions1)
	# Step 2: init Plane Game
	GluttonousSnake = game.GameState()
	# Step 3: play game
	# Step 3.1: obtain init state
	action0 = np.array([0, 0, 0, 1])  # [1,0,0]do nothing,[0,1,0]left,[0,0,1]right
	action01 = np.array([0, 0, 0, 1])
	observation0, reward0, terminal,score,reward01,score1  = GluttonousSnake.frame_step(action0,action01)
	observation0 = cv2.cvtColor(cv2.resize(observation0, (80, 80)), cv2.COLOR_BGR2GRAY)
	ret, observation0 = cv2.threshold(observation0,1,255,cv2.THRESH_BINARY)
	# cv2.imwrite("3.jpg",observation0,[int(cv2.IMWRITE_JPEG_QUALITY),5])

	brain.setInitState(observation0)
	brain1.setInitState(observation0)

	aa = []
	bb = []


	# Step 3.2: run the game
	while 1!= 0:
		action = brain.getAction()
		action1 = brain1.getAction()
		nextObservation,reward,terminal,score,reward1,score1 = GluttonousSnake.frame_step(action,action1)
		#nextObservation, reward, terminal, score, reward1, score1 = GluttonousSnake.frame_step1(action1)
		nextObservation = preprocess(nextObservation)
		brain.setPerception(nextObservation,  action,  reward,  terminal)
		brain1.setPerception(nextObservation, action1, reward1, terminal)

		if score > top:
			top = score
			numm = brain.cull()
			aa.append(numm)

		if score1 > top1:
			top1 = score1
			num = brain1.cull()
			bb.append(num)

		print('top1:%u' % top)
		print('top2:%u' % top1)
		print(aa)
		print(bb)
def main():
	playSnake()

if __name__ == '__main__':
	main()