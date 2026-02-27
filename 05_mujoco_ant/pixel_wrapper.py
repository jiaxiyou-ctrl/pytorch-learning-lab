"""Pixel observation wrapper: converts state obs to camera images for vision-based RL."""

import collections

import cv2
import gymnasium as gym
import numpy as np

class PixelObsWrapper(gym.Wrapper):
    """Wraps a MuJoCo env to return stacked pixel frames instead of state vectors."""

    def __init__(
        self,
        env: gym.Env, 
        image_size: int = 84,
        frame_stack: int = 3,
    ) -> None:
        super().__init__(env)
        self.image_size = image_size
        self.frame_stack = frame_stack
        
        #Frame buffer
        self.frames = collections.deque(maxlen=frame_stack)

        #Observation space
        num_channels = frame_stack * 3
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1.0,
            shape=(num_channels, image_size, image_size),
            dtype=np.float32
        )

    def _get_image(self) -> np.ndarray:

        #Capture the image from the envoronment and preprocess it

        raw_image = self.env.render()
        resized = cv2.resize(
            raw_image, 
            (self.image_size, self.image_size), 
            interpolation=cv2.INTER_AREA
        )

        normalized = resized.astype(np.float32) / 255.0

        transposed = normalized.transpose(2, 0, 1)

        return transposed

    def reset(self, **kwargs):

        """reset the environment and fill the frame buffer"""
        _obs, info = super().reset(**kwargs)

        initial_frame = self._get_image()

        for _ in range(self.frame_stack):
            self.frames.append(initial_frame)
        
        staked_obs = self._get_stacked_obs()
        return staked_obs, info

    def step(self, action):
        """take a step in the environment and update the frame buffer"""

        _obs, reward, terminated, truncated, info = self.env.step(action)

        new_frame = self._get_image()
        self.frames.append(new_frame)

        staked_obs = self._get_stacked_obs()
        return staked_obs, reward, terminated, truncated, info

    def _get_stacked_obs(self) -> np.ndarray:
        """Concatentate all frames in the buffer along the channel dimension"""

        return np.concatenate(list(self.frames), axis=0)




