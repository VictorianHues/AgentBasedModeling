# Climate behaviour ABM with coupled environmental dynamics

This repository contains the agent-based model implementation and experiments supporting the corresponding
report into individual climate-related decision-making in coupled human-environment dynamical systems.

<p align="center">
    <img src="docs/source/figures/11b_env.gif" width="70%" /><br>
    <em>Fig 1: Emergence of clustering environment level across time.</em>
</p>

<p align="center">
    <img src="docs/source/figures/phase_portraits.png" width="70%"/><br>
    <em>Fig 2: Mean field phase portraits for different agent rationality levels and social norm weight.</em>
</p>


To contribute to this project, or reproduce our results, see the project documentation available at 
[victorianhues.github.io/AgentBasedModeling](https://victorianhues.github.io/AgentBasedModeling/).

## Quick reference
- **Installation**: Clone the repository and install the required dependencies using `uv sync` (if you have uv) or `pip install -r requirements.txt`. It is recommended to use `uv` for managing the environment.
- **Running the model**: Use `make -j` to run all the scripts and get approximate (but faster) results. Alternatively, you can run `quality=HIGH make -j` to get more accurate results. 
- **Contributing**: Follow the guidelines in the [documentation](https://victorianhues.github.io/AgentBasedModeling/contributing.html).

## Authors

Victoria Peterson - 15476758

Karolina Chłopicka - 15716546

Henry Zwart - 15393879

Shania Sinha - 14379031


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
