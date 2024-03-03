import cv2, imageio, json, os
import numpy as np
from pyquaternion import Quaternion


from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


def _make_instruction(base_path):
    if 'Door_Closing' in base_path:
        return "Close the door."
    if 'Door_Opening' in base_path:
        return "Open the door."
    if 'Drawer_Closing' in base_path:
        return "Close the drawer."
    if 'Drawer_Opening' in base_path:
        return "Open the drawer."
    if 'Handle' in base_path:
        return "Grasp the handle."
    if 'Pick_and_Place' in base_path:
        return "Pick the object and place it."
    if "Button" in base_path:
        return "Switch the button."
    print(base_path, "task not found!")
    raise NotImplementedError


def _resize(fr, size=(256, 256)):
    if fr.shape[:2] == size: # most images are already (256, 256)
        return fr
    fr = cv2.resize(fr, size, interpolation=cv2.INTER_AREA)
    return fr


def _to_quat(quat_vector):
    x, y, z, w = quat_vector
    return Quaternion(w=w, x=x, y=y, z=z)


def _rot_state(q):
    q = _to_quat(q)
    yaw, pitch, roll = q.yaw_pitch_roll
    return np.array([roll, pitch, yaw])


def _delta_quat(q1, q2):
    q1, q2 = _to_quat(q1), _to_quat(q2)
    return q1.inverse * q2

def _delta_rot(q1, q2):
    q_rot = _delta_quat(q1, q2)
    yaw, pitch, roll = q_rot.yaw_pitch_roll
    return np.array([roll, pitch, yaw])

def _grip_state(gripper):
    return np.array([gripper]).astype(np.float32)


class ExampleDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'wrist_image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Stick wrist camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Full stick state: [x,y,z,roll,pitch,yaw,gripper]',
                        ),
                        'gripper': tfds.features.Tensor(
                            shape=(1,),
                            dtype=np.float32,
                            doc='gripper state',
                        ),
                        'xyz': tfds.features.Tensor(
                            shape=(3,),
                            dtype=np.float32,
                            doc='xyz stick state',
                        ),
                        'rot': tfds.features.Tensor(
                            shape=(3,),
                            dtype=np.float32,
                            doc='Rotation state as [roll,pitch,yaw]',
                        ),
                        'quat': tfds.features.Tensor(
                            shape=(4,),
                            dtype=np.float32,
                            doc='Robot rotation quaternion [x,y,z,w]',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Stick action: [dx,dy,dz,droll,dptich,dyaw,gripper (absolute)]',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='/scratch2/sudeep/homes_of_new_york/*/*/*/*'),
            # 'val': self._generate_examples(path='data/val/episode_*.npy'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        trajs = glob.glob(path)
        trajs = [t for t in trajs if "Random" not in t]
        print(f'got {len(trajs)} records')

        def _parse_example(episode_path):
            reader = imageio.get_reader(os.path.join(episode_path, 'compressed_video_h264.mp4'))
            with open(os.path.join(episode_path, 'labels.json'), 'r') as f:
                data = json.load(f)

            episode = []
            for i, fr in enumerate(reader):
                if str(i + 1) not in data:
                    break
                fr = _resize(fr)
                obs, next_obs = data[str(i)], data[str(i + 1)]

                # compute the state vector
                xyz = np.array(obs['xyz']).astype(np.float32)
                quat_vec = np.array(obs['quats']).astype(np.float32)
                rot_vec  = _rot_state(obs['quats']).astype(np.float32)
                grip_state = _grip_state(obs['gripper']).astype(np.float32)
                state = np.concatenate((xyz, rot_vec, grip_state)).astype(np.float32)

                # compute the action vector
                delta_xyz = np.array(next_obs['xyz']) - np.array(obs['xyz'])
                delta_rot = _delta_rot(obs['quats'], next_obs['quats'])
                action = np.concatenate((delta_xyz, delta_rot, _grip_state(next_obs['gripper']))).astype(np.float32)

                language_inst = _make_instruction(episode_path)
                language_embedding = self._embed([language_inst])[0].numpy()

                episode.append({
                    'observation': {
                        'wrist_image': fr,
                        'state': state,
                        'gripper': grip_state,
                        'xyz': xyz,
                        'rot': rot_vec,
                        'quat': quat_vec,
                    },
                    'action': action,
                    'discount': 1.0,
                    'reward': 1,  # data is optimal
                    'is_first': i == 0,
                    'is_last': str(i + 2) not in data,
                    'is_terminal': str(i + 2) not in data,
                    'language_instruction': language_inst,
                    'language_embedding': language_embedding,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample


        # for smallish datasets, use single-thread parsing
        for sample in trajs:
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

