import cProfile
from pstats import SortKey, Stats

from abm_project.utils import piecewise_exponential_update
from abm_project.vectorised_model import VectorisedModel


def main():
    """
    Profile vectorised model
    """
    num_steps = 10000
    model = VectorisedModel(
        num_agents=10000,
        width=100,
        height=100,
        env_update_fn=piecewise_exponential_update(recovery=1, pollution=1, gamma=0.01),
        neighb_prediction_option=None,
        max_storage=num_steps,
    )
    model.run(num_steps)


if __name__ == "__main__":
    with cProfile.Profile() as profile:
        main()
        Stats(profile).strip_dirs().sort_stats(SortKey.TIME).print_stats(10)
