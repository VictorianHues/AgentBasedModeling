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

FOURIER_POWER_HEATMAPS = \
		fourier_power_rationality_vs_gamma_s_nonadaptive_nonpredictive_peerconst_$(QUALITY)_quality.pdf \
		fourier_power_rationality_vs_gamma_s_nonadaptive_predictive_peerconst_$(QUALITY)_quality.pdf \
		fourier_power_rationality_vs_gamma_s_nonadaptive_predictive_peerrandomised_$(QUALITY)_quality.pdf \
		fourier_power_rationality_vs_gamma_s_adaptive_nonpredictive_peerrandomised_$(QUALITY)_quality.pdf

EQ_ENV_VARYING_GAMMA_S_HEATMAPS = \
		equilibrium_env_rationality_vs_gamma_s_nonpredictive_peerconst_$(QUALITY)_quality.pdf \
		equilibrium_env_rationality_vs_gamma_s_nonpredictive_peerrandomised_$(QUALITY)_quality.pdf \
		equilibrium_env_rationality_vs_gamma_s_predictive_peerconst_$(QUALITY)_quality.pdf \
		equilibrium_env_rationality_vs_gamma_s_predictive_peerrandomised_$(QUALITY)_quality.pdf \

EQ_ENV_VARYING_MEMORY_HEATMAPS = \
		equilibrium_env_rationality_vs_memory_gamma_s_0_004_$(QUALITY)_quality.pdf \
		equilibrium_env_rationality_vs_memory_gamma_s_0_01_$(QUALITY)_quality.pdf
		

FIGURE_NAMES = \
		sensitivity_analysis_outcome_distributions_$(QUALITY)_quality.pdf \
		equilibrium_env_vs_rationality_$(QUALITY)_quality.pdf \
		equilibrium_env_vs_gamma_s_$(QUALITY)_quality.pdf \
		$(FOURIER_POWER_HEATMAPS) \
		$(EQ_ENV_VARYING_GAMMA_S_HEATMAPS) \
		$(EQ_ENV_VARYING_MEMORY_HEATMAPS) \
		time_series_mean_env_by_rationality_$(QUALITY)_quality.pdf \
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

$(FIGURES_DIR)/equilibrium_env_vs_rationality_$(QUALITY)_quality.pdf: \
			scripts/plot_equilibrium_env_for_varying_rationality.py \
			$(DATA_DIR)/eq_env_vs_rationality_$(QUALITY)_quality.npz \
			| $(FIGURES_DIR)
	$(ENTRYPOINT) $< $(QUALITY_PARAMS)

$(FIGURES_DIR)/equilibrium_env_vs_gamma_s_$(QUALITY)_quality.pdf: \
			scripts/plot_eq_environment_for_varying_agent_adaptation.py \
			$(DATA_DIR)/eq_env_vs_gamma_s_$(QUALITY)_quality.npz \
			| $(FIGURES_DIR)
	$(ENTRYPOINT) $< $(QUALITY_PARAMS)

$(FIGURES_DIR)/time_series_mean_env_by_rationality_$(QUALITY)_quality.pdf: \
			scripts/plot_time_series_env_varying_rationality.py \
			$(DATA_DIR)/time_series_mean_env_for_varying_rationality_$(QUALITY)_quality.npz \
			| $(FIGURES_DIR)
	$(ENTRYPOINT) $< $(QUALITY_PARAMS)


# Equilibrium environment for varying memory size and rationality
$(FIGURES_DIR)/equilibrium_env_rationality_vs_memory_gamma_s_%_$(QUALITY)_quality.pdf: \
			scripts/environment_vs_rationality_and_memory_phaseplot.py \
			$(DATA_DIR)/equilibrium_env_rationality_vs_memory_gamma_s_%_$(QUALITY)_quality.npz \
			| $(FIGURES_DIR)
	$(ENTRYPOINT) $< $(QUALITY_PARAMS) --gamma-s '$*'

#  ----------------------
#  Fourier power heatmaps
#  ----------------------
#  Non-adaptive, non-predictive, constant peer pressure
$(FIGURES_DIR)/fourier_power_rationality_vs_gamma_s_nonadaptive_nonpredictive_peerconst_$(QUALITY)_quality.pdf: \
			scripts/plot_fourier_power_varying_gamma_s_rationality.py \
			$(DATA_DIR)/fourier_power_rationality_vs_gamma_s_nonadaptive_nonpredictive_peerconst_$(QUALITY)_quality.npz \
			| $(FIGURES_DIR)
	$(ENTRYPOINT) $< $(QUALITY_PARAMS) --const-peer-pressure


#  Non-adaptive, linear-predictive, constant peer pressure
$(FIGURES_DIR)/fourier_power_rationality_vs_gamma_s_nonadaptive_predictive_peerconst_$(QUALITY)_quality.pdf: \
			scripts/plot_fourier_power_varying_gamma_s_rationality.py \
			$(DATA_DIR)/fourier_power_rationality_vs_gamma_s_nonadaptive_predictive_peerconst_$(QUALITY)_quality.npz \
			| $(FIGURES_DIR)
	$(ENTRYPOINT) $< $(QUALITY_PARAMS) --const-peer-pressure --neighborhood-prediction linear


#  Non-adaptive, linear-predictive, randomised peer pressure
$(FIGURES_DIR)/fourier_power_rationality_vs_gamma_s_nonadaptive_predictive_peerrandomised_$(QUALITY)_quality.pdf: \
			scripts/plot_fourier_power_varying_gamma_s_rationality.py \
			$(DATA_DIR)/fourier_power_rationality_vs_gamma_s_nonadaptive_predictive_peerrandomised_$(QUALITY)_quality.npz \
			| $(FIGURES_DIR)
	$(ENTRYPOINT) $< $(QUALITY_PARAMS) --neighborhood-prediction linear


#  Adaptive, non-predictive, randomised peer pressure
$(FIGURES_DIR)/fourier_power_rationality_vs_gamma_s_adaptive_nonpredictive_peerrandomised_$(QUALITY)_quality.pdf: \
			scripts/plot_fourier_power_varying_gamma_s_rationality.py \
			$(DATA_DIR)/fourier_power_rationality_vs_gamma_s_adaptive_nonpredictive_peerrandomised_$(QUALITY)_quality.npz \
			| $(FIGURES_DIR)
	$(ENTRYPOINT) $< $(QUALITY_PARAMS) --dynamic-action-preference


#  ----------------------
#  Equilibrium environment heatmaps for varying gamma_s and rationality
#  ----------------------
# Non-predictive, constant peer pressure
$(FIGURES_DIR)/equilibrium_env_rationality_vs_gamma_s_nonpredictive_peerconst_$(QUALITY)_quality.pdf: \
			scripts/plot_environment_varying_gamma_s_rationality.py \
			$(DATA_DIR)/equilibrium_env_rationality_vs_gamma_s_nonpredictive_peerconst_$(QUALITY)_quality.npz  \
			| $(FIGURES_DIR)
	$(ENTRYPOINT) $< $(QUALITY_PARAMS) --const-peer-pressure


# Non-predictive, randomised peer pressure
$(FIGURES_DIR)/equilibrium_env_rationality_vs_gamma_s_nonpredictive_peerrandomised_$(QUALITY)_quality.pdf: \
			scripts/plot_environment_varying_gamma_s_rationality.py \
			$(DATA_DIR)/equilibrium_env_rationality_vs_gamma_s_nonpredictive_peerrandomised_$(QUALITY)_quality.npz  \
			| $(FIGURES_DIR)
	$(ENTRYPOINT) $< $(QUALITY_PARAMS) 


# Predictive, constant peer pressure
$(FIGURES_DIR)/equilibrium_env_rationality_vs_gamma_s_predictive_peerconst_$(QUALITY)_quality.pdf: \
			scripts/plot_environment_varying_gamma_s_rationality.py \
			$(DATA_DIR)/equilibrium_env_rationality_vs_gamma_s_predictive_peerconst_$(QUALITY)_quality.npz  \
			| $(FIGURES_DIR)
	$(ENTRYPOINT) $< $(QUALITY_PARAMS) --const-peer-pressure --neighborhood-prediction linear


# Predictive, randomised peer pressure
$(FIGURES_DIR)/equilibrium_env_rationality_vs_gamma_s_predictive_peerrandomised_$(QUALITY)_quality.pdf: \
			scripts/plot_environment_varying_gamma_s_rationality.py \
			$(DATA_DIR)/equilibrium_env_rationality_vs_gamma_s_predictive_peerrandomised_$(QUALITY)_quality.npz  \
			| $(FIGURES_DIR)
	$(ENTRYPOINT) $< $(QUALITY_PARAMS) --neighborhood-prediction linear

# ========================
# Data and heavy analysis
# ========================
$(DATA_DIR)/eq_env_vs_rationality_$(QUALITY)_quality.npz: \
			scripts/measure_eq_environment_for_varying_rationality.py \
			| $(DATA_DIR)
	$(ENTRYPOINT) $< $(QUALITY_PARAMS)


$(DATA_DIR)/eq_env_vs_gamma_s_$(QUALITY)_quality.npz: \
			scripts/measure_eq_environment_for_varying_agent_adaptation.py \
			| $(DATA_DIR)
	$(ENTRYPOINT) $< $(QUALITY_PARAMS)


$(DATA_DIR)/equilibrium_env_rationality_vs_memory_gamma_s_%_$(QUALITY)_quality.npz: \
			scripts/equilibrium_env_vs_memory_and_rationality_measurements.py \
			| $(DATA_DIR)
	$(ENTRYPOINT) $< $(QUALITY_PARAMS) --gamma-s '$*'


$(DATA_DIR)/time_series_mean_env_for_varying_rationality_$(QUALITY)_quality.npz: \
			scripts/measure_time_series_env_varying_rationality.py \
			| $(DATA_DIR)
	$(ENTRYPOINT) $< $(QUALITY_PARAMS)

	 
$(DATA_DIR)/sensitivity_analysis_outcome_measurements_$(QUALITY)_quality.npz: \
			scripts/sensitivity_analysis_outcome_measurements.py \
			| $(DATA_DIR)
	$(ENTRYPOINT) $< $(QUALITY_PARAMS) 


#  ----------------------
#  Fourier power heatmaps
#  ----------------------
#  Non-adaptive, non-predictive, constant peer pressure
$(DATA_DIR)/fourier_power_rationality_vs_gamma_s_nonadaptive_nonpredictive_peerconst_$(QUALITY)_quality.npz: \
			scripts/measure_fourier_power_varying_gamma_s.py \
			| $(DATA_DIR)
	$(ENTRYPOINT) $< $(QUALITY_PARAMS) --const-peer-pressure

#
#  Non-adaptive, linear-predictive, constant peer pressure
$(DATA_DIR)/fourier_power_rationality_vs_gamma_s_nonadaptive_predictive_peerconst_$(QUALITY)_quality.npz: \
			scripts/measure_fourier_power_varying_gamma_s.py \
			| $(DATA_DIR)
	$(ENTRYPOINT) $< $(QUALITY_PARAMS) --const-peer-pressure --neighborhood-prediction linear


#  Non-adaptive, linear-predictive, randomised peer pressure
$(DATA_DIR)/fourier_power_rationality_vs_gamma_s_nonadaptive_predictive_peerrandomised_$(QUALITY)_quality.npz: \
			scripts/measure_fourier_power_varying_gamma_s.py \
			| $(DATA_DIR)
	$(ENTRYPOINT) $< $(QUALITY_PARAMS) --neighborhood-prediction linear

	
#  Adaptive, non-predictive, randomised peer pressure
$(DATA_DIR)/fourier_power_rationality_vs_gamma_s_adaptive_nonpredictive_peerrandomised_$(QUALITY)_quality.npz: \
			scripts/measure_fourier_power_varying_gamma_s.py \
			| $(DATA_DIR)
	$(ENTRYPOINT) $< $(QUALITY_PARAMS) --dynamic-action-preference


#  ----------------------
#  Equilibrium environment heatmaps for varying gamma_s and rationality
#  ----------------------
# Non-predictive, constant peer pressure
$(DATA_DIR)/equilibrium_env_rationality_vs_gamma_s_nonpredictive_peerconst_$(QUALITY)_quality.npz: \
			scripts/measure_environment_varying_gamma_s_rationality.py \
			| $(DATA_DIR)
	$(ENTRYPOINT) $< $(QUALITY_PARAMS) --const-peer-pressure


# Non-predictive, randomised peer pressure
$(DATA_DIR)/equilibrium_env_rationality_vs_gamma_s_nonpredictive_peerrandomised_$(QUALITY)_quality.npz: \
			scripts/measure_environment_varying_gamma_s_rationality.py \
			| $(DATA_DIR)
	$(ENTRYPOINT) $< $(QUALITY_PARAMS) 


# Predictive, constant peer pressure
$(DATA_DIR)/equilibrium_env_rationality_vs_gamma_s_predictive_peerconst_$(QUALITY)_quality.npz: \
			scripts/measure_environment_varying_gamma_s_rationality.py \
			| $(DATA_DIR)
	$(ENTRYPOINT) $< $(QUALITY_PARAMS) --const-peer-pressure --neighborhood-prediction linear


# Predictive, randomised peer pressure
$(DATA_DIR)/equilibrium_env_rationality_vs_gamma_s_predictive_peerrandomised_$(QUALITY)_quality.npz: \
			scripts/measure_environment_varying_gamma_s_rationality.py \
			| $(DATA_DIR)
	$(ENTRYPOINT) $< $(QUALITY_PARAMS) --neighborhood-prediction linear 


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


