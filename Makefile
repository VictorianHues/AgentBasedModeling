FIGURES_DIR = results/figures
ANIMATIONS_DIR = results/animations
DATA_DIR = data

ENTRYPOINT ?= uv run
QUALITY ?= low
ABM_MAX_WORKERS ?=
ABM_CHUNKSIZE ?=

ifneq ($(strip $(ABM_MAX_WORKERS)),)
export ABM_MAX_WORKERS
endif

ifneq ($(strip $(ABM_CHUNKSIZE)),)
export ABM_CHUNKSIZE
endif

# Temporal autocorrelation/susceptibility sweep controls.
VARIANCE_LAST_K ?= 5
ACF_MAX_LAG ?= 200
SLICE_LAMBDA ?= 4.0
ACF_LAMBDA ?= $(SLICE_LAMBDA)
ACF_GAMMA ?= 0.0090
NEIGHBORHOOD_PREDICTION ?= none
CONST_PEER ?= 0

# CCDF / spatial-correlation report figure controls.
CCDF_LAMBDA ?= 4.0
CCDF_GAMMA_S ?= 0.0091
CCDF_NEIGHBORHOOD_PREDICTION ?= none
CCDF_CONST_PEER ?= 0
CCDF_REPEATS ?=
CORR_WINDOW ?= 1
CCDF_ACF_MAX_LAG ?= 200
CCDF_ENV_UPDATE ?= piecewise
CCDF_MEMORY_COUNT ?= 10
CCDF_SEED ?= 42

ifeq ($(QUALITY),low)
	QUALITY_PARAMS = --quick
	GAMMA_S_STEPS ?= 10
	RATIONALITY_STEPS ?= 10
else ifeq ($(QUALITY),high)
	QUALITY_PARAMS = 
	GAMMA_S_STEPS ?= 25
	RATIONALITY_STEPS ?= 25
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
		equilibrium_env_vs_gamma_s_peerrandomised_$(QUALITY)_quality.pdf \
		equilibrium_env_vs_gamma_s_peerconst_$(QUALITY)_quality.pdf \
		fourier_power_vs_gamma_s_peerrandomised_$(QUALITY)_quality.pdf \
		fourier_power_vs_gamma_s_peerconst_$(QUALITY)_quality.pdf \
		$(FOURIER_POWER_HEATMAPS) \
		$(EQ_ENV_VARYING_GAMMA_S_HEATMAPS) \
		$(EQ_ENV_VARYING_MEMORY_HEATMAPS) \
		time_series_mean_env_by_rationality_$(QUALITY)_quality.pdf \
		appendix_phase_portraits.pdf \
		appendix_fixed_point_mean_action.pdf \
		num_clusters_lambda4.00_gamma0.005_piecewise_ccdf_$(QUALITY)_quality.pdf

# Prefix figure names with figures directory
FIGURES = $(patsubst %, $(FIGURES_DIR)/%, $(FIGURE_NAMES))

DATA_ARTIFACT_NAMES = \
		sensitivity_analysis_outcome_measurements_$(QUALITY)_quality.npz \
		eq_env_vs_rationality_$(QUALITY)_quality.npz \
		eq_env_vs_gamma_s_peerrandomised_$(QUALITY)_quality.npz \
		eq_env_vs_gamma_s_peerconst_$(QUALITY)_quality.npz \
		fourier_power_vs_gamma_s_peerrandomised_$(QUALITY)_quality.npz \
		fourier_power_vs_gamma_s_peerconst_$(QUALITY)_quality.npz \
		fourier_power_rationality_vs_gamma_s_nonadaptive_nonpredictive_peerconst_$(QUALITY)_quality.npz \
		fourier_power_rationality_vs_gamma_s_nonadaptive_predictive_peerconst_$(QUALITY)_quality.npz \
		fourier_power_rationality_vs_gamma_s_nonadaptive_predictive_peerrandomised_$(QUALITY)_quality.npz \
		fourier_power_rationality_vs_gamma_s_adaptive_nonpredictive_peerrandomised_$(QUALITY)_quality.npz \
		equilibrium_env_rationality_vs_gamma_s_nonpredictive_peerconst_$(QUALITY)_quality.npz \
		equilibrium_env_rationality_vs_gamma_s_nonpredictive_peerrandomised_$(QUALITY)_quality.npz \
		equilibrium_env_rationality_vs_gamma_s_predictive_peerconst_$(QUALITY)_quality.npz \
		equilibrium_env_rationality_vs_gamma_s_predictive_peerrandomised_$(QUALITY)_quality.npz \
		equilibrium_env_rationality_vs_memory_gamma_s_0_004_$(QUALITY)_quality.npz \
		equilibrium_env_rationality_vs_memory_gamma_s_0_01_$(QUALITY)_quality.npz \
		time_series_mean_env_for_varying_rationality_$(QUALITY)_quality.npz

DATA_ARTIFACTS = $(patsubst %, $(DATA_DIR)/%, $(DATA_ARTIFACT_NAMES))

.PHONY: all help measure-all plot-all documentation clean clean-all temporal-autocorr-susceptibility powerlaw-spatial-corr-param powerlaw-spatial-corr-gamma-sweep powerlaw-spatial-corr-gamma-0041 powerlaw-spatial-corr-gamma-0045 powerlaw-spatial-corr-gamma-0049 phase-transition-analysis phase-transition-all-configurations phase-transition-nonpredictive-peerrandomised phase-transition-nonpredictive-peerconst phase-transition-predictive-peerrandomised phase-transition-predictive-peerconst



# ========================
# Figures and animations
# ========================
all: $(FIGURES) $(ANIMATIONS) powerlaw-spatial-corr-gamma-sweep

measure-all: $(DATA_ARTIFACTS)

plot-all: $(FIGURES)

help:
	@echo Common workflows:
	@echo   make -j  (includes CCDF gamma sweep)
	@echo   QUALITY=high make -j
	@echo   make measure-all QUALITY=high
	@echo   make plot-all QUALITY=high
	@echo   make phase-transition-analysis
	@echo   make phase-transition-all-configurations
	@echo   make powerlaw-spatial-corr-gamma-sweep
	@echo.
	@echo Data-only experiment runs:
	@echo   make measure-all
	@echo   make data/eq_env_vs_rationality_low_quality.npz
	@echo   make data/fourier_power_rationality_vs_gamma_s_nonadaptive_predictive_peerconst_low_quality.npz
	@echo.
	@echo Plot-only workflows from existing data:
	@echo   make plot-all
	@echo   make results/figures/equilibrium_env_vs_rationality_low_quality.pdf
	@echo   make results/figures/fourier_power_vs_gamma_s_peerconst_low_quality.pdf
	@echo.
	@echo Inspect or debug commands without executing:
	@echo   make -n TARGET_NAME
	@echo.
	@echo Temporal autocorr/susceptibility knobs:
	@echo   QUALITY=low_or_high CONST_PEER=0_or_1 NEIGHBORHOOD_PREDICTION=none_or_linear
	@echo   GAMMA_S_STEPS=integer RATIONALITY_STEPS=integer VARIANCE_LAST_K=integer
	@echo   ACF_MAX_LAG=integer SLICE_LAMBDA=float ACF_LAMBDA=float ACF_GAMMA=float
	@echo.
	@echo CCDF/spatial-correlation knobs:
	@echo   CCDF_LAMBDA=float CCDF_GAMMA_S=float
	@echo   CCDF_NEIGHBORHOOD_PREDICTION=none_or_linear CCDF_CONST_PEER=0_or_1
	@echo   CORR_WINDOW=integer CCDF_ACF_MAX_LAG=integer CCDF_ENV_UPDATE=linear_or_piecewise
	@echo   CCDF_MEMORY_COUNT=integer CCDF_SEED=integer CCDF_REPEATS=integer
	@echo.
	@echo Parallelism knobs for simulation workers:
	@echo   ABM_MAX_WORKERS=integer ABM_CHUNKSIZE=integer

temporal-autocorr-susceptibility: | $(DATA_DIR) $(FIGURES_DIR)
	$(ENTRYPOINT) scripts/measure_experiments.py heatmap --metric environment \
		$(QUALITY_PARAMS) --dynamic-action-preference $(if $(filter 1,$(CONST_PEER)),--const-peer-pressure,) $(if $(filter none,$(NEIGHBORHOOD_PREDICTION)),,--neighborhood-prediction $(NEIGHBORHOOD_PREDICTION)) \
		--gamma-s-steps $(GAMMA_S_STEPS) --rationality-steps $(RATIONALITY_STEPS) \
		--variance-last-k $(VARIANCE_LAST_K) --acf-max-lag $(ACF_MAX_LAG)
	$(ENTRYPOINT) scripts/plot_experiments.py heatmap --metric environment \
		$(QUALITY_PARAMS) --dynamic-action-preference $(if $(filter 1,$(CONST_PEER)),--const-peer-pressure,) $(if $(filter none,$(NEIGHBORHOOD_PREDICTION)),,--neighborhood-prediction $(NEIGHBORHOOD_PREDICTION)) \
		--variance-slice-lambda $(SLICE_LAMBDA) --autocorr-slice-lambda $(SLICE_LAMBDA) \
		--acf-lambda $(ACF_LAMBDA) --acf-gamma $(ACF_GAMMA)

powerlaw-spatial-corr-param: | $(FIGURES_DIR)
	$(ENTRYPOINT) scripts/create_ccdfs.py $(QUALITY_PARAMS) \
		--lambda-value $(CCDF_LAMBDA) --gamma-s $(CCDF_GAMMA_S) \
		$(if $(filter none,$(CCDF_NEIGHBORHOOD_PREDICTION)),--neighborhood-prediction none,--neighborhood-prediction $(CCDF_NEIGHBORHOOD_PREDICTION)) $(if $(filter 1,$(CCDF_CONST_PEER)),--const-peer-pressure,) \
		--corr-window $(CORR_WINDOW) --acf-max-lag $(CCDF_ACF_MAX_LAG) \
		--env-update-type $(CCDF_ENV_UPDATE) --memory-count $(CCDF_MEMORY_COUNT) \
		--seed $(CCDF_SEED) $(if $(strip $(CCDF_REPEATS)),--repeats $(CCDF_REPEATS),)

powerlaw-spatial-corr-gamma-sweep: \
	powerlaw-spatial-corr-gamma-0041 \
	powerlaw-spatial-corr-gamma-0045 \
	powerlaw-spatial-corr-gamma-0049

powerlaw-spatial-corr-gamma-0041:
	$(MAKE) powerlaw-spatial-corr-param CCDF_GAMMA_S=0.0041

powerlaw-spatial-corr-gamma-0045:
	$(MAKE) powerlaw-spatial-corr-param CCDF_GAMMA_S=0.0045

powerlaw-spatial-corr-gamma-0049:
	$(MAKE) powerlaw-spatial-corr-param CCDF_GAMMA_S=0.0049

phase-transition-analysis: temporal-autocorr-susceptibility powerlaw-spatial-corr-param

phase-transition-all-configurations: \
	phase-transition-nonpredictive-peerrandomised \
	phase-transition-nonpredictive-peerconst \
	phase-transition-predictive-peerrandomised \
	phase-transition-predictive-peerconst

phase-transition-nonpredictive-peerrandomised:
	$(MAKE) phase-transition-analysis \
		CONST_PEER=0 NEIGHBORHOOD_PREDICTION=none \
		CCDF_CONST_PEER=0 CCDF_NEIGHBORHOOD_PREDICTION=none

phase-transition-nonpredictive-peerconst:
	$(MAKE) phase-transition-analysis \
		CONST_PEER=1 NEIGHBORHOOD_PREDICTION=none \
		CCDF_CONST_PEER=1 CCDF_NEIGHBORHOOD_PREDICTION=none

phase-transition-predictive-peerrandomised:
	$(MAKE) phase-transition-analysis \
		CONST_PEER=0 NEIGHBORHOOD_PREDICTION=linear \
		CCDF_CONST_PEER=0 CCDF_NEIGHBORHOOD_PREDICTION=linear

phase-transition-predictive-peerconst:
	$(MAKE) phase-transition-analysis \
		CONST_PEER=1 NEIGHBORHOOD_PREDICTION=linear \
		CCDF_CONST_PEER=1 CCDF_NEIGHBORHOOD_PREDICTION=linear

$(FIGURES_DIR)/appendix_phase_portraits.pdf: \
			scripts/appendix_phase_portraits.py \
			src/abm_project/plotting.py \
			| $(FIGURES_DIR) # Ensure figures directory exists
	$(ENTRYPOINT) $< # Note: this symbol refers to the first dependency, i.e., the script

$(FIGURES_DIR)/appendix_fixed_point_mean_action.pdf: \
			scripts/appendix_fixed_point_actions.py \
			| $(FIGURES_DIR) 
	$(ENTRYPOINT) $< 

$(FIGURES_DIR)/num_clusters_lambda4.00_gamma0.005_piecewise_ccdf_$(QUALITY)_quality.pdf: \
			scripts/create_ccdfs.py \
			| $(FIGURES_DIR)
	$(ENTRYPOINT) $< $(QUALITY_PARAMS)

$(FIGURES_DIR)/sensitivity_analysis_outcome_distributions_$(QUALITY)_quality.pdf: \
			scripts/sensitivity_analysis_plots.py \
			src/abm_project/sensitivity_analysis.py \
			$(DATA_DIR)/sensitivity_analysis_outcome_measurements_$(QUALITY)_quality.npz \
			| $(FIGURES_DIR)
	$(ENTRYPOINT) $< $(QUALITY_PARAMS)

$(FIGURES_DIR)/equilibrium_env_vs_rationality_$(QUALITY)_quality.pdf: \
			scripts/plot_experiments.py \
			$(DATA_DIR)/eq_env_vs_rationality_$(QUALITY)_quality.npz \
			| $(FIGURES_DIR)
	$(ENTRYPOINT) $< curve --metric eq-env-vs-rationality $(QUALITY_PARAMS)

$(FIGURES_DIR)/equilibrium_env_vs_gamma_s_peerrandomised_$(QUALITY)_quality.pdf: \
			scripts/plot_experiments.py \
			$(DATA_DIR)/eq_env_vs_gamma_s_peerrandomised_$(QUALITY)_quality.npz \
			| $(FIGURES_DIR)
	$(ENTRYPOINT) $< curve --metric eq-env-vs-gamma $(QUALITY_PARAMS)


$(FIGURES_DIR)/equilibrium_env_vs_gamma_s_peerconst_$(QUALITY)_quality.pdf: \
			scripts/plot_experiments.py \
			$(DATA_DIR)/eq_env_vs_gamma_s_peerconst_$(QUALITY)_quality.npz \
			| $(FIGURES_DIR)
	$(ENTRYPOINT) $< curve --metric eq-env-vs-gamma $(QUALITY_PARAMS) --const-peer-pressure


$(FIGURES_DIR)/fourier_power_vs_gamma_s_peerrandomised_$(QUALITY)_quality.pdf: \
			scripts/plot_experiments.py \
			$(DATA_DIR)/fourier_power_vs_gamma_s_peerrandomised_$(QUALITY)_quality.npz \
			| $(FIGURES_DIR)
	$(ENTRYPOINT) $< curve --metric fourier-vs-gamma $(QUALITY_PARAMS)


$(FIGURES_DIR)/fourier_power_vs_gamma_s_peerconst_$(QUALITY)_quality.pdf: \
			scripts/plot_experiments.py \
			$(DATA_DIR)/fourier_power_vs_gamma_s_peerconst_$(QUALITY)_quality.npz \
			| $(FIGURES_DIR)
	$(ENTRYPOINT) $< curve --metric fourier-vs-gamma $(QUALITY_PARAMS) --const-peer-pressure

$(FIGURES_DIR)/time_series_mean_env_by_rationality_$(QUALITY)_quality.pdf: \
			scripts/plot_experiments.py \
			$(DATA_DIR)/time_series_mean_env_for_varying_rationality_$(QUALITY)_quality.npz \
			| $(FIGURES_DIR)
	$(ENTRYPOINT) $< timeseries --metric env-vs-rationality $(QUALITY_PARAMS)


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
			scripts/plot_experiments.py \
			$(DATA_DIR)/fourier_power_rationality_vs_gamma_s_nonadaptive_nonpredictive_peerconst_$(QUALITY)_quality.npz \
			| $(FIGURES_DIR)
	$(ENTRYPOINT) $< heatmap --metric fourier $(QUALITY_PARAMS) --const-peer-pressure


#  Non-adaptive, linear-predictive, constant peer pressure
$(FIGURES_DIR)/fourier_power_rationality_vs_gamma_s_nonadaptive_predictive_peerconst_$(QUALITY)_quality.pdf: \
			scripts/plot_experiments.py \
			$(DATA_DIR)/fourier_power_rationality_vs_gamma_s_nonadaptive_predictive_peerconst_$(QUALITY)_quality.npz \
			| $(FIGURES_DIR)
	$(ENTRYPOINT) $< heatmap --metric fourier $(QUALITY_PARAMS) --const-peer-pressure --neighborhood-prediction linear


#  Non-adaptive, linear-predictive, randomised peer pressure
$(FIGURES_DIR)/fourier_power_rationality_vs_gamma_s_nonadaptive_predictive_peerrandomised_$(QUALITY)_quality.pdf: \
			scripts/plot_experiments.py \
			$(DATA_DIR)/fourier_power_rationality_vs_gamma_s_nonadaptive_predictive_peerrandomised_$(QUALITY)_quality.npz \
			| $(FIGURES_DIR)
	$(ENTRYPOINT) $< heatmap --metric fourier $(QUALITY_PARAMS) --neighborhood-prediction linear


#  Adaptive, non-predictive, randomised peer pressure
$(FIGURES_DIR)/fourier_power_rationality_vs_gamma_s_adaptive_nonpredictive_peerrandomised_$(QUALITY)_quality.pdf: \
			scripts/plot_experiments.py \
			$(DATA_DIR)/fourier_power_rationality_vs_gamma_s_adaptive_nonpredictive_peerrandomised_$(QUALITY)_quality.npz \
			| $(FIGURES_DIR)
	$(ENTRYPOINT) $< heatmap --metric fourier $(QUALITY_PARAMS) --dynamic-action-preference


#  ----------------------
#  Equilibrium environment heatmaps for varying gamma_s and rationality
#  ----------------------
# Non-predictive, constant peer pressure
$(FIGURES_DIR)/equilibrium_env_rationality_vs_gamma_s_nonpredictive_peerconst_$(QUALITY)_quality.pdf: \
			scripts/plot_experiments.py \
			$(DATA_DIR)/equilibrium_env_rationality_vs_gamma_s_nonpredictive_peerconst_$(QUALITY)_quality.npz  \
			| $(FIGURES_DIR)
	$(ENTRYPOINT) $< heatmap --metric environment $(QUALITY_PARAMS) --dynamic-action-preference --const-peer-pressure


# Non-predictive, randomised peer pressure
$(FIGURES_DIR)/equilibrium_env_rationality_vs_gamma_s_nonpredictive_peerrandomised_$(QUALITY)_quality.pdf: \
			scripts/plot_experiments.py \
			$(DATA_DIR)/equilibrium_env_rationality_vs_gamma_s_nonpredictive_peerrandomised_$(QUALITY)_quality.npz  \
			| $(FIGURES_DIR)
	$(ENTRYPOINT) $< heatmap --metric environment $(QUALITY_PARAMS) --dynamic-action-preference


# Predictive, constant peer pressure
$(FIGURES_DIR)/equilibrium_env_rationality_vs_gamma_s_predictive_peerconst_$(QUALITY)_quality.pdf: \
			scripts/plot_experiments.py \
			$(DATA_DIR)/equilibrium_env_rationality_vs_gamma_s_predictive_peerconst_$(QUALITY)_quality.npz  \
			| $(FIGURES_DIR)
	$(ENTRYPOINT) $< heatmap --metric environment $(QUALITY_PARAMS) --dynamic-action-preference --const-peer-pressure --neighborhood-prediction linear


# Predictive, randomised peer pressure
$(FIGURES_DIR)/equilibrium_env_rationality_vs_gamma_s_predictive_peerrandomised_$(QUALITY)_quality.pdf: \
			scripts/plot_experiments.py \
			$(DATA_DIR)/equilibrium_env_rationality_vs_gamma_s_predictive_peerrandomised_$(QUALITY)_quality.npz  \
			| $(FIGURES_DIR)
	$(ENTRYPOINT) $< heatmap --metric environment $(QUALITY_PARAMS) --dynamic-action-preference --neighborhood-prediction linear

# ========================
# Data and heavy analysis
# ========================
$(DATA_DIR)/eq_env_vs_rationality_$(QUALITY)_quality.npz: \
			scripts/measure_experiments.py \
			| $(DATA_DIR)
	$(ENTRYPOINT) $< curve --metric eq-env-vs-rationality $(QUALITY_PARAMS)


$(DATA_DIR)/eq_env_vs_gamma_s_peerrandomised_$(QUALITY)_quality.npz: \
			scripts/measure_experiments.py \
			| $(DATA_DIR)
	$(ENTRYPOINT) $< curve --metric eq-env-vs-gamma $(QUALITY_PARAMS)


$(DATA_DIR)/eq_env_vs_gamma_s_peerconst_$(QUALITY)_quality.npz: \
			scripts/measure_experiments.py \
			| $(DATA_DIR)
	$(ENTRYPOINT) $< curve --metric eq-env-vs-gamma $(QUALITY_PARAMS) --const-peer-pressure


$(DATA_DIR)/fourier_power_vs_gamma_s_peerrandomised_$(QUALITY)_quality.npz: \
			scripts/measure_experiments.py \
			| $(DATA_DIR)
	$(ENTRYPOINT) $< curve --metric fourier-vs-gamma $(QUALITY_PARAMS)


$(DATA_DIR)/fourier_power_vs_gamma_s_peerconst_$(QUALITY)_quality.npz: \
			scripts/measure_experiments.py \
			| $(DATA_DIR)
	$(ENTRYPOINT) $< curve --metric fourier-vs-gamma $(QUALITY_PARAMS) --const-peer-pressure


$(DATA_DIR)/equilibrium_env_rationality_vs_memory_gamma_s_%_$(QUALITY)_quality.npz: \
			scripts/equilibrium_env_vs_memory_and_rationality_measurements.py \
			| $(DATA_DIR)
	$(ENTRYPOINT) $< $(QUALITY_PARAMS) --gamma-s '$*'


$(DATA_DIR)/time_series_mean_env_for_varying_rationality_$(QUALITY)_quality.npz: \
			scripts/measure_experiments.py \
			| $(DATA_DIR)
	$(ENTRYPOINT) $< timeseries --metric env-vs-rationality $(QUALITY_PARAMS)

	 
$(DATA_DIR)/sensitivity_analysis_outcome_measurements_$(QUALITY)_quality.npz: \
			scripts/sensitivity_analysis_outcome_measurements.py \
			| $(DATA_DIR)
	$(ENTRYPOINT) $< $(QUALITY_PARAMS) 


#  ----------------------
#  Fourier power heatmaps
#  ----------------------
#  Non-adaptive, non-predictive, constant peer pressure
$(DATA_DIR)/fourier_power_rationality_vs_gamma_s_nonadaptive_nonpredictive_peerconst_$(QUALITY)_quality.npz: \
			scripts/measure_experiments.py \
			| $(DATA_DIR)
	$(ENTRYPOINT) $< heatmap --metric fourier $(QUALITY_PARAMS) --const-peer-pressure

#
#  Non-adaptive, linear-predictive, constant peer pressure
$(DATA_DIR)/fourier_power_rationality_vs_gamma_s_nonadaptive_predictive_peerconst_$(QUALITY)_quality.npz: \
			scripts/measure_experiments.py \
			| $(DATA_DIR)
	$(ENTRYPOINT) $< heatmap --metric fourier $(QUALITY_PARAMS) --const-peer-pressure --neighborhood-prediction linear


#  Non-adaptive, linear-predictive, randomised peer pressure
$(DATA_DIR)/fourier_power_rationality_vs_gamma_s_nonadaptive_predictive_peerrandomised_$(QUALITY)_quality.npz: \
			scripts/measure_experiments.py \
			| $(DATA_DIR)
	$(ENTRYPOINT) $< heatmap --metric fourier $(QUALITY_PARAMS) --neighborhood-prediction linear

	
#  Adaptive, non-predictive, randomised peer pressure
$(DATA_DIR)/fourier_power_rationality_vs_gamma_s_adaptive_nonpredictive_peerrandomised_$(QUALITY)_quality.npz: \
			scripts/measure_experiments.py \
			| $(DATA_DIR)
	$(ENTRYPOINT) $< heatmap --metric fourier $(QUALITY_PARAMS) --dynamic-action-preference


#  ----------------------
#  Equilibrium environment heatmaps for varying gamma_s and rationality
#  ----------------------
# Non-predictive, constant peer pressure
$(DATA_DIR)/equilibrium_env_rationality_vs_gamma_s_nonpredictive_peerconst_$(QUALITY)_quality.npz: \
			scripts/measure_experiments.py \
			| $(DATA_DIR)
	$(ENTRYPOINT) $< heatmap --metric environment $(QUALITY_PARAMS) --dynamic-action-preference --const-peer-pressure


# Non-predictive, randomised peer pressure
$(DATA_DIR)/equilibrium_env_rationality_vs_gamma_s_nonpredictive_peerrandomised_$(QUALITY)_quality.npz: \
			scripts/measure_experiments.py \
			| $(DATA_DIR)
	$(ENTRYPOINT) $< heatmap --metric environment $(QUALITY_PARAMS) --dynamic-action-preference


# Predictive, constant peer pressure
$(DATA_DIR)/equilibrium_env_rationality_vs_gamma_s_predictive_peerconst_$(QUALITY)_quality.npz: \
			scripts/measure_experiments.py \
			| $(DATA_DIR)
	$(ENTRYPOINT) $< heatmap --metric environment $(QUALITY_PARAMS) --dynamic-action-preference --const-peer-pressure --neighborhood-prediction linear


# Predictive, randomised peer pressure
$(DATA_DIR)/equilibrium_env_rationality_vs_gamma_s_predictive_peerrandomised_$(QUALITY)_quality.npz: \
			scripts/measure_experiments.py \
			| $(DATA_DIR)
	$(ENTRYPOINT) $< heatmap --metric environment $(QUALITY_PARAMS) --dynamic-action-preference --neighborhood-prediction linear


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


