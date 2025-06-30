FIGURES_DIR = results/figures
ANIMATIONS_DIR = results/animations
DATA_DIR = data

ENTRYPOINT ?= uv run
QUALITY ?= low

ifeq ($(QUALITY),low)
	QUALITY_PARAMS = --quick
else ifeq ($(QUALITY),high)
	QUALITY_PARAMS = 
else
	$(error Invalid quality specifier: $(QUALITY). Choose 'low' or 'high'.)
endif

FIGURE_NAMES = \
		sensitivity_analysis_outcome_distributions_$(QUALITY)_quality.pdf \
		phaseplot_env_vs_rationality_memory_$(QUALITY)_quality.pdf \
		appendix_phase_portraits.pdf \
		appendix_fixed_point_mean_action.pdf

# Prefix figure names with figures directory
FIGURES = $(patsubst %, $(FIGURES_DIR)/%, $(FIGURE_NAMES))

.PHONY: all documentation clean



# ========================
# Figures and animations
# ========================
all: $(FIGURES) $(ANIMATIONS)

$(FIGURES_DIR)/appendix_phase_portraits.pdf: \
			scripts/appendix_phase_portraits.py \
			src/abm_project/plotting.py \
			| $(FIGURES_DIR) # Ensure figures directory exists
	$(ENTRYPOINT) $< # Note: this symbol refers to the first dependency, i.e., the script

$(FIGURES_DIR)/appendix_fixed_point_mean_action.pdf: \
			scripts/appendix_fixed_point_actions.py \
			| $(FIGURES_DIR) 
	$(ENTRYPOINT) $< 

$(FIGURES_DIR)/sensitivity_analysis_outcome_distributions_$(QUALITY)_quality.pdf: \
			scripts/sensitivity_analysis_plots.py \
			src/abm_project/sensitivity_analysis.py \
			$(DATA_DIR)/sensitivity_analysis_outcome_measurements_$(QUALITY)_quality.npz \
			| $(FIGURES_DIR)
	$(ENTRYPOINT) $< $(QUALITY_PARAMS)


$(FIGURES_DIR)/phaseplot_env_vs_rationality_memory_$(QUALITY)_quality.pdf: \
			scripts/environment_vs_rationality_and_memory_phaseplot.py \
			$(DATA_DIR)/eq_env_vs_rationality_and_memory_$(QUALITY)_quality.npz \
			| $(FIGURES_DIR)
	$(ENTRYPOINT) $< $(QUALITY_PARAMS)


# ========================
# Data and heavy analysis
# ========================
$(DATA_DIR)/eq_env_vs_rationality_and_memory_$(QUALITY)_quality.npz: \
			scripts/equilibrium_env_vs_memory_and_rationality_measurements.py \
			| $(DATA_DIR)
	$(ENTRYPOINT) $< $(QUALITY_PARAMS)
	 
$(DATA_DIR)/sensitivity_analysis_outcome_measurements_$(QUALITY)_quality.npz: \
			scripts/sensitivity_analysis_outcome_measurements.py \
			| $(DATA_DIR)
	$(ENTRYPOINT) $< $(QUALITY_PARAMS) 





# ========================
# Create required directories
# ========================
$(FIGURES_DIR):
	mkdir -p $@

$(ANIMATIONS_DIR):
	mkdir -p $@

$(DATA_DIR):
	mkdir -p $@


# ========================
# Utilities
# ========================

documentation:
	uv run --group docs python -m sphinx -M html docs/source docs/build

clean:
	rm -rf $(FIGURES_DIR) $(ANIMATIONS_DIR)

clean-all:
	rm -rf $(FIGURES_DIR) $(ANIMATIONS_DIR) $(DATA_DIR)


