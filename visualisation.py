import matplotlib.pyplot as plt
import numpy as np

def create_every_state(N):
  states = np.zeros((N*N, N*N))
  states[np.arange(N*N), np.arange(N*N)] = 1.
  states = states.reshape(N*N, N, N)
  return states

def visualize_actions(Q_values):
  N = int(np.sqrt(Q_values.shape[0]))
  matrix = Q_values.reshape(N, N, 2)
  plt.matshow(matrix.argmax(axis=-1))
  plt.axis('off')
  for i1 in range(N + 1):
    plt.axhline(i1 - 0.5, color='black', linewidth=2)

  for j in range(N + 1):
    plt.axvline(j - 0.5, color='black', linewidth=2)

def visualize_ensemble_actions(Q_values, action_mapping=None):
  N = int(np.sqrt(Q_values.shape[1]))
  
  if action_mapping is None:
    action_mapping = np.ones((N, N))

  Q_values = np.where(action_mapping.reshape(1, -1, 1), Q_values[:, :, :], Q_values[:, :, ::-1])
  matrix = Q_values.reshape(Q_values.shape[0], int(np.sqrt(Q_values.shape[1])), int(np.sqrt(Q_values.shape[1])), 2).argmax(axis=-1)
  fig, axes = plt.subplots(1, Q_values.shape[0])
  for i in range(Q_values.shape[0]):
    axes[i].matshow(matrix[i])
    axes[i].axis('off')
    for i1 in range(N + 1):
      axes[i].axhline(i1 - 0.5, color='black', linewidth=2)

    for j in range(N + 1):
      axes[i].axvline(j - 0.5, color='black', linewidth=2)
  fig.set_size_inches(30, 10)

def visualize_Q_values(Q_values, action_mapping=None):
  N = int(np.sqrt(Q_values.shape[1]))
  if action_mapping is None:
    action_mapping = np.ones((N, N))

  Q_values = np.where(action_mapping.reshape(1, -1, 1), Q_values[:, :, :], Q_values[:, :, ::-1])

  matrix = Q_values.reshape(Q_values.shape[0], int(np.sqrt(Q_values.shape[1])), int(np.sqrt(Q_values.shape[1])) * 2)
  fig, axes = plt.subplots(1, Q_values.shape[0])
  for i in range(Q_values.shape[0]):
    axes[i].matshow(matrix[i], aspect=2)
    axes[i].axis('off')
    

    for i1 in range(N + 1):
      axes[i].axhline(i1 - 0.5, color='black', linewidth=2)

    for j in range(N + 1):
      axes[i].axvline(2 * j - 0.5, color='black', linewidth=2)
  fig.set_size_inches(30, 10)


def visualize_uncertainty(Q_values, action_mapping=None):
  N = int(np.sqrt(Q_values.shape[1]))
  if action_mapping is None:
    action_mapping = np.ones((N, N))

  Q_values = np.where(action_mapping.reshape(1, -1, 1), Q_values[:, :, :], Q_values[:, :, ::-1])

  uncertainty = np.std(Q_values, axis=0)
  matrix = uncertainty.reshape(int(np.sqrt(Q_values.shape[1])), int(np.sqrt(Q_values.shape[1])) * 2)
  fig, axes = plt.subplots(1, 1)
  plot = axes.matshow(matrix, aspect=2)
  axes.axis('off')
  for i1 in range(N + 1):
    axes.axhline(i1 - 0.5, color='black', linewidth=2)

  for j in range(N + 1):
    axes.axvline(2 * j - 0.5, color='black', linewidth=2)
  fig.set_size_inches(10, 10)
  fig.colorbar(plot, shrink=0.7)