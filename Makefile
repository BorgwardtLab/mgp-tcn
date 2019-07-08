MAKEFLAGS = --warn-undefined-variables

CONFIG_FILES_PATH = configs
EXPERIMENTS_PATH = experiments

CONFIG_FILES = $(wildcard $(CONFIG_FILES_PATH)/*.json)
CONFIGS = $(foreach FILE,$(CONFIG_FILES),$(basename $(notdir $(FILE))))

EXPERIMENT_TARGETS = $(foreach CONFIG,$(CONFIGS),$(EXPERIMENTS_PATH)/$(CONFIG))

MYSQL_DATABASE := mimic
MYSQL_USER := <Username>


.PHONY = run_experiments query

all: query run_experiments

mock_data_results:
	python3 -m code.mgp_tcn.test_mgp_tcn -F $@
	@mv $@/1/* $@
	@rm -r $@/1

query:
	echo Runnung query...
	cd code/query && ./main.sh $(MYSQL_DATABASE) $(MYSQL_USER)

run_experiments: $(EXPERIMENT_TARGETS)
	echo $(CONFIG_FILES)
	echo $(CONFIGS)
	echo $<

$(EXPERIMENTS_PATH)/%: $(CONFIG_FILES_PATH)/%.json
	@echo Running config file $<
	@-mkdir -p $@
	python3 -m code.mgp_tcn.mgp_tcn_fit -p -F $@ with $<
	@mv $@/1/* $@
	@rm -r $@/1
