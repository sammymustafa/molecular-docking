from setup import *

# HELPERS
class MLP(object):
  """A random two layer neural network."""

  def __init__(self, in_dim, out_dim):
    self.w1 = np.random.randn(in_dim, out_dim)
    self.b1 = np.random.randn(out_dim)
    self.w2 = np.random.randn(out_dim, out_dim)
    self.b2 = np.random.randn(out_dim)

  def forward(self, h):
    h_new = np.dot(h, self.w1) + self.b1
    h_new = h_new * (h_new > 0)
    h_new = np.dot(h_new, self.w2) + self.b2
    return h_new

  def __call__(self, h):
    return self.forward(h)


def apply_transform(R, t, x):
  """Apply a rotation and translation to a set of points."""
  return np.dot(x, R.T) + t


class EGNN(object):

  def __init__(self, dim):
    """Random weight initialization."""
    self.message_mlp = MLP(dim * 3 + 1, dim)
    self.feature_mlp = MLP(dim * 2, dim)
    self.coords_mlp = MLP(dim, 1)

  def forward(self, h, e, x):
    """Apply an EGNN layer.

    Parameters
    ----------
    h:
      node features as a numpy array of shape (N, D)
    e:
      edge features as a numpy array of shape (N, N, D)
    x:
      set of points in 3D as a numpy array of shape (N, 3)

    Returns
    -------
    h_new:
      new node features as a numpy array of shape (N, D)
    x_new:
      new set of points in 3D as a numpy array of shape (N, 3)

    """

    # Expand h to get all pairs of h's, your final dimension should be (N, N, 2 * D)
    N, D = h.shape
    h_i = np.repeat(h[:, np.newaxis, :], N, axis=1)
    h_j = np.repeat(h[np.newaxis, :, :], N, axis=0)
    h_pairs = np.concatenate([h_i, h_j], axis=-1)

    # Get all x_i - x_j vectors (you can just broadcast, without repeat)
    x_i = np.repeat(x[:, np.newaxis, :], N, axis=1)
    x_j = np.repeat(x[np.newaxis, :, :], N, axis=0)
    x_ij = x_i - x_j

    # Compute the squared norm of the x_ij vectors
    norm_x_ij = np.linalg.norm(x_ij, axis=-1, keepdims=True)**2

    # Concatenate h pairs, distances, and edge features
    m_input = np.concatenate([h_pairs, norm_x_ij, e], axis=-1)

    # Compute messages for all i, j pairs
    m_ij = self.message_mlp(m_input.reshape(N * N, -1)).reshape(N, N, -1)

    # Aggregate messages per node (sum across j)
    m_i = np.sum(m_ij, axis=1)

    # Update features
    h_new = self.feature_mlp(np.concatenate([h, m_i], axis=-1))

    # Update coordinates
    x_ij_norm = np.linalg.norm(x_ij, axis=-1, keepdims=True)
    x_ij_unit = x_ij / np.where(x_ij_norm > 0, x_ij_norm, 1)
    coord_updates = self.coords_mlp(m_i) * x_ij_unit
    x_new = x + np.sum(coord_updates, axis=1)

    return h_new, x_new


def test_equivariance():
  # Set parameters
  N = 64
  D = 32
  h = np.random.randn(N, D)
  e = np.random.randn(N, N, D)
  x = np.random.randn(N, 3)
  egnn = EGNN(D)

  # Sample a random roto-translation
  R = Rotation.random().as_matrix()
  t = np.random.randn(3)
  x_rot = apply_transform(R, t, x)

  # Run model on x
  _, x_new = egnn.forward(h, e, x)
  x_new_rot = apply_transform(R, t, x_new)

  # Run the model on x_rot
  _, x_rot_new = egnn.forward(h, e, x_rot)

  # Compare the outputs
  if np.linalg.norm(x_new_rot - x_rot_new, axis=-1).max() < 1e-4:
    print("Equivariance test passed!")
  else:
    print("Equivariance test failed!")


test_equivariance()