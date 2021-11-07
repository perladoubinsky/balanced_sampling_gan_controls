import os.path
import argparse
import numpy as np

from utils.logger import setup_logger
from utils_balancing import train_boundary

def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser(
      description='Train semantic boundary with given latent codes and '
                  'attribute scores.')
  parser.add_argument('-o', '--output_dir', type=str, required=True,
                      help='Directory to save the output results. (required)')
  parser.add_argument('-c', '--latent_codes_path', type=str, required=True,
                      help='Path to the input latent codes. (required)')
  parser.add_argument('-s', '--scores_path', type=str, required=True,
                      help='Path to the dictionary containing all the attributes scores. (required)')
  parser.add_argument('-a', '--attribute', type=str,
                     help='Attribute for which to compute the boundary (required).')
  parser.add_argument('-n', '--num_samples_boundary', type=int,
                     help='Number of samples to use to compute the boundary (required).')
  parser.add_argument('-t', '--confidence_threshold', type=float, default=None,
                     help='Confidence threshold to filter ambiguous samples.')
  parser.add_argument('--boundary_name', type=str, default='boundary.npy')

  return parser.parse_args()


def main():
  """Main function."""
  args = parse_args()
  logger = setup_logger(args.output_dir, logger_name='generate_data')

  logger.info('Loading latent codes.')
  if not os.path.isfile(args.latent_codes_path):
    raise ValueError(f'Latent codes `{args.latent_codes_path}` does not exist!')
  latent_codes = np.load(args.latent_codes_path)

  logger.info('Loading attribute scores.')
  if not os.path.isfile(args.scores_path):
    raise ValueError(f'Attribute scores `{args.scores_path}` does not exist!')
  scores_dict = np.load(args.scores_path, allow_pickle=True)[()]

  boundary = train_boundary(latent_codes=latent_codes,
                            scores_dict=scores_dict,
                            attribute=args.attribute,
                            confidence_t=args.confidence_threshold,
                            n=args.num_samples_boundary)
    
  np.save(os.path.join(args.output_dir, args.boundary_name), boundary)

if __name__ == '__main__':
  main()
