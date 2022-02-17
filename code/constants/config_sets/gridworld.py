# This file constructs configuration sets for easier management
# A predefined option can be run as follows (1000 for Pong with seed 0, for instance):
# bc_base.py --env-key ALE-Pong --config-set 1000 --seed 0

# ID Structure: <Method Identifier + Subversion> & <Teacher Level> & <Budget>
# [Example] Early Advising with 100k budget and competent teacher: 20011
#
# <Teacher Level>
# 0: Incompetent
# 1: Competent
#
# <Budget>
# 0: 25000
# 1: 100000
# 2: 12500

def generate_config():
    config = {0: {}, 1: {}}
    config[0]['run-id'] = None
    config[0]['process-index'] = 0
    config[0]['machine-name'] = 'NONE'
    config[0]['n-training-frames'] = int(100e3)
    config[0]['n-evaluation-trials'] = 10
    config[0]['evaluation-period'] = int(500)

    config[1]['visualize-videos'] = True
    config[0]['evaluation-visualization-period'] = 5
    config[0]['visualization-period'] = 1000

    config[1]['save-obs-images'] = False

    config[0]['dqn-type'] = 'egreedy'  # 'egreedy', 'noisy'
    config[0]['dqn-gamma'] = 0.99
    config[0]['dqn-rm-type'] = 'uniform'
    config[0]['dqn-rm-init'] = int(500)
    config[0]['dqn-rm-max'] = int(5000)
    config[0]['dqn-per-alpha'] = 0.4
    config[0]['dqn-per-beta'] = 0.6
    config[1]['dqn-per-ims'] = True
    config[0]['dqn-target-update'] = 100
    config[0]['dqn-batch-size'] = 32
    config[0]['dqn-learning-rate'] = 0.0001
    config[0]['dqn-train-per-step'] = 1
    config[0]['dqn-train-period'] = 2
    config[0]['dqn-adam-eps'] = 0.00015
    config[0]['dqn-eps-start'] = 1.0
    config[0]['dqn-eps-final'] = 0.01
    config[0]['dqn-eps-steps'] = 5000
    config[0]['dqn-huber-loss-delta'] = 1.0
    config[0]['dqn-n-hidden-layers'] = 1
    config[0]['dqn-hidden-size-1'] = 512
    config[0]['dqn-hidden-size-2'] = 0
    config[1]['dqn-dueling'] = True

    config[1]['dqn-dropout'] = False
    config[0]['dqn-dropout-rate'] = 0.2
    config[0]['dqn-dropout-uc-ensembles'] = int(100)

    # Twin DQN is to be supervised trained with the original DQNs samples and targets - has dropout enabled by default
    # for uncertainty estimations
    config[1]['dqn-twin'] = False
    config[0]['dqn-twin-dropout-rate'] = 0.2
    config[0]['dqn-twin-dropout-uc-ensembles'] = int(100)
    config[0]['dqn-twin-learning-rate'] = 0.0001
    config[0]['dqn-twin-adam-eps'] = 0.00015
    config[0]['dqn-twin-huber-loss-delta'] = 1.0
    config[0]['dqn-twin-n-hidden-layers'] = 1
    config[0]['dqn-twin-hidden-size-1'] = 512
    config[0]['dqn-twin-hidden-size-2'] = 0
    config[0]['dqn-twin-uncertainty-type'] = 0  # 0: Q-values variance, 1: Best Q-values occurrences

    config[1]['dump-replay-memory'] = False
    config[1]['use-gpu'] = True
    config[1]['save-models'] = False
    config[0]['model-save-period'] = int(200e3)
    config[0]['env-key'] = 'NONE'
    config[0]['env-training-seed'] = 0
    config[0]['env-evaluation-seed'] = 1
    config[0]['seed'] = 0

    config[1]['load-teacher'] = False
    config[0]['teacher-level'] = 1  # 0: Incompetent, 1: Competent
    config[1]['execute-teacher-policy'] = False  # Only for debugging the teacher policy

    # Pre-collected Dataset
    config[1]['load-demonstrations-dataset'] = False
    config[1]['preserve-initial-demonstrations-dataset'] = False

    # Learning from Demonstration
    config[1]['use-lfd-loss'] = False
    config[0]['n-pretraining-iterations'] = 0

    config[0]['dqn-ql-loss-weight'] = 1.0
    config[0]['dqn-lm-loss-weight'] = 0.2
    config[0]['dqn-l2-loss-weight'] = 1e-5
    config[0]['dqn-lm-loss-margin'] = 0.8

    # Collection
    config[0]['advice-collection-budget'] = 0
    config[1]['mistake-correction-mode'] = False

    # 'none'
    # 'early'
    # 'random'
    # 'student_model_uc' (uncertainty)
    # 'teacher_model_uc' (uncertainty)
    config[0]['advice-collection-method'] = 'none'

    # For 'student_model_uncertainty' driven collection:
    config[0]['student-model-uc-th'] = 0.01
    config[1]['use-proportional-student-model-uc-th'] = False  # Use proportionally determined threshold
    config[0]['proportional-student-model-uc-th-window-size'] = 0
    config[0]['proportional-student-model-uc-th-window-size-min'] = 0  # number of uc values to be collected until
    # the percentile is computed
    config[0]['proportional-student-model-uc-th-percentile'] = 0

    # For 'teacher_model_uncertainty' driven collection:
    config[0]['teacher-model-uc-th'] = 0  # Used both in Collection and Reuse decisions
    config[1]['autoset-teacher-model-uc-th'] = False

    config[1]['preserve-collected-advice'] = False  # Always preserve collected advice transitions in replay memory

    # Imitation
    config[0]['advice-imitation-method'] = 'none'  # 'none', 'periodic'
    config[0]['advice-imitation-period-steps'] = 0
    config[0]['advice-imitation-period-samples'] = 0
    config[0]['advice-imitation-training-iterations-init'] = 0
    config[0]['advice-imitation-training-iterations-periodic'] = 0

    config[0]['bc-batch-size'] = 32
    config[0]['bc-learning-rate'] = 0.0001
    config[0]['bc-adam-eps'] = 0.00015
    config[0]['bc-dropout-rate'] = 0.35
    config[0]['bc-hidden-size'] = int(512)
    config[0]['bc-uc-ensembles'] = int(100)

    # Reuse (based on the teacher imitation model)
    config[0]['advice-reuse-method'] = 'none'  # 'none', 'restricted', 'extended'
    config[0]['advice-reuse-probability'] = 0
    config[1]['advice-reuse-probability-decay'] = False
    config[0]['advice-reuse-probability-decay-begin'] = 0
    config[0]['advice-reuse-probability-decay-end'] = 0
    config[0]['advice-reuse-probability-final'] = 0

    config[1]['evaluate-advice-reuse-model'] = False

    # Advice reuse stopping conditions
    config[1]['advice-reuse-stopping'] = False
    config[0]['advice-reuse-stopping-eval-start'] = 50
    config[0]['advice-reuse-stopping-eval-window-size'] = 5
    config[0]['advice-reuse-stopping-eval-proximity'] = 0.9

    return config


# ======================================================================================================================

BUDGET_OPTIONS = [
    10,     # 0
    25,     # 1
    50,     # 2
    100,    # 3
    250,    # 4
    500,    # 5
    1000,   # 6
    5000,   # 7
    10000,  # 8
    100000, # 9
]
TEACHER_LEVEL_OPTIONS = [0]

CONFIG_SETS = {}

# ----------------------------------------------------------------------------------------------------------------------
# Generate demonstrator (Long training)

id = 0
CONFIG_SETS[id] = generate_config()
CONFIG_SETS[id][0]['dqn-rm-type'] = 'uniform'
CONFIG_SETS[id][1]['save-models'] = True
CONFIG_SETS[id][0]['evaluation-period'] = int(50e3)
CONFIG_SETS[id][0]['n-evaluation-trials'] = 10
CONFIG_SETS[id][0]['n-training-frames'] = int(10e6)



# ----------------------------------------------------------------------------------------------------------------------
# Uncertainty test setups

id = 5
CONFIG_SETS[id] = generate_config()
CONFIG_SETS[id][1]['dqn-twin'] = True
CONFIG_SETS[id][0]['dqn-twin-uncertainty-type'] = 0  # Q-values variances

id = 6
CONFIG_SETS[id] = generate_config()
CONFIG_SETS[id][1]['dqn-twin'] = True
CONFIG_SETS[id][0]['dqn-twin-uncertainty-type'] = 1  # Best Q-values occurrences

# ----------------------------------------------------------------------------------------------------------------------
# Evaluate teacher (with a single evaluation step)

for i, teacher_level_option in enumerate(TEACHER_LEVEL_OPTIONS):
    id = int('50' + str(i))
    CONFIG_SETS[id] = generate_config()
    CONFIG_SETS[id][1]['load-teacher'] = True
    CONFIG_SETS[id][0]['teacher-level'] = teacher_level_option
    CONFIG_SETS[id][1]['execute-teacher-policy'] = True
    CONFIG_SETS[id][0]['dqn-rm-init'] = int(1e6)  # Do NOT train the model in this evaluation setup
    CONFIG_SETS[id][0]['n-training-frames'] = int(50e3)
    CONFIG_SETS[id][1]['visualize-videos'] = True
    CONFIG_SETS[id][0]['evaluation-visualization-period'] = 1

# ----------------------------------------------------------------------------------------------------------------------
# NA: No Advising (Training from scratch)

id = 1000
CONFIG_SETS[id] = generate_config()

CONFIG_SETS[id][1]['dqn-twin'] = True
CONFIG_SETS[id][0]['dqn-twin-uncertainty-type'] = 0  # Q-values variances

# ----------------------------------------------------------------------------------------------------------------------
# EA: Early Advising

for i, teacher_level_option in enumerate(TEACHER_LEVEL_OPTIONS):
    for j, budget_option in enumerate(BUDGET_OPTIONS):
        id = int('200' + str(i) + str(j))
        CONFIG_SETS[id] = generate_config()
        CONFIG_SETS[id][1]['load-teacher'] = True
        CONFIG_SETS[id][0]['teacher-level'] = teacher_level_option
        CONFIG_SETS[id][0]['advice-collection-budget'] = budget_option
        CONFIG_SETS[id][0]['advice-collection-method'] = 'early'

        CONFIG_SETS[id][1]['dqn-twin'] = True
        CONFIG_SETS[id][0]['dqn-twin-uncertainty-type'] = 0  # Q-values variances

# ----------------------------------------------------------------------------------------------------------------------
# RA: Random Advising

for i, teacher_level_option in enumerate(TEACHER_LEVEL_OPTIONS):
    for j, budget_option in enumerate(BUDGET_OPTIONS):
        id = int('210' + str(i) + str(j))
        CONFIG_SETS[id] = generate_config()
        CONFIG_SETS[id][1]['load-teacher'] = True
        CONFIG_SETS[id][0]['teacher-level'] = teacher_level_option
        CONFIG_SETS[id][0]['advice-collection-budget'] = budget_option
        CONFIG_SETS[id][0]['advice-collection-method'] = 'random'

# ----------------------------------------------------------------------------------------------------------------------
# EA: Early Advising with Cheating (Mistake Correction)

for i, teacher_level_option in enumerate(TEACHER_LEVEL_OPTIONS):
    for j, budget_option in enumerate(BUDGET_OPTIONS):
        id = int('220' + str(i) + str(j))
        CONFIG_SETS[id] = generate_config()
        CONFIG_SETS[id][1]['load-teacher'] = True
        CONFIG_SETS[id][0]['teacher-level'] = teacher_level_option
        CONFIG_SETS[id][0]['advice-collection-budget'] = budget_option
        CONFIG_SETS[id][0]['advice-collection-method'] = 'early'
        CONFIG_SETS[id][1]['mistake-correction-mode'] = True

# ----------------------------------------------------------------------------------------------------------------------
# AR: Advice Reuse
# Paper: "Action Advising with Advice Imitation in Deep Reinforcement Learning" (https://arxiv.org/abs/2104.08441)

for i, teacher_level_option in enumerate(TEACHER_LEVEL_OPTIONS):
    for j, budget_option in enumerate(BUDGET_OPTIONS):
        id = int('300' + str(i) + str(j))
        CONFIG_SETS[id] = generate_config()
        CONFIG_SETS[id][1]['load-teacher'] = True
        CONFIG_SETS[id][0]['teacher-level'] = teacher_level_option
        CONFIG_SETS[id][0]['advice-collection-budget'] = budget_option
        CONFIG_SETS[id][0]['advice-collection-method'] = 'early'
        CONFIG_SETS[id][0]['teacher-model-uc-th'] = 0.01
        CONFIG_SETS[id][0]['advice-imitation-method'] = 'periodic'
        CONFIG_SETS[id][0]['advice-imitation-period-steps'] = int(1e9)
        CONFIG_SETS[id][0]['advice-imitation-period-samples'] = CONFIG_SETS[id][0]['advice-collection-budget']
        CONFIG_SETS[id][0]['advice-imitation-training-iterations-init'] = int(200e3)
        CONFIG_SETS[id][0]['advice-reuse-method'] = 'restricted'
        CONFIG_SETS[id][0]['advice-reuse-probability'] = 0.5

# ----------------------------------------------------------------------------------------------------------------------
# AR+A: AR is enhanced with the automatic threshold tuning technique

for i, teacher_level_option in enumerate(TEACHER_LEVEL_OPTIONS):
    for j, budget_option in enumerate(BUDGET_OPTIONS):
        id = int('310' + str(i) + str(j))
        CONFIG_SETS[id] = generate_config()
        CONFIG_SETS[id][1]['load-teacher'] = True
        CONFIG_SETS[id][0]['teacher-level'] = teacher_level_option
        CONFIG_SETS[id][0]['advice-collection-budget'] = budget_option
        CONFIG_SETS[id][0]['advice-collection-method'] = 'early'
        CONFIG_SETS[id][0]['advice-imitation-method'] = 'periodic'
        CONFIG_SETS[id][0]['advice-imitation-period-steps'] = int(1e9)
        CONFIG_SETS[id][0]['advice-imitation-period-samples'] = CONFIG_SETS[id][0]['advice-collection-budget']
        CONFIG_SETS[id][0]['advice-imitation-training-iterations-init'] = int(200e3)
        CONFIG_SETS[id][0]['advice-reuse-method'] = 'restricted'
        CONFIG_SETS[id][0]['advice-reuse-probability'] = 0.5
        CONFIG_SETS[id][1]['autoset-teacher-model-uc-th'] = True

# ----------------------------------------------------------------------------------------------------------------------
# AR+A+E: AR+A is enhanced with the unrestricted reuse procedure

for i, teacher_level_option in enumerate(TEACHER_LEVEL_OPTIONS):
    for j, budget_option in enumerate(BUDGET_OPTIONS):
        id = int('320' + str(i) + str(j))
        CONFIG_SETS[id] = generate_config()
        CONFIG_SETS[id][1]['load-teacher'] = True
        CONFIG_SETS[id][0]['teacher-level'] = teacher_level_option
        CONFIG_SETS[id][0]['advice-collection-budget'] = budget_option
        CONFIG_SETS[id][0]['advice-collection-method'] = 'early'
        CONFIG_SETS[id][0]['advice-imitation-method'] = 'periodic'
        CONFIG_SETS[id][0]['advice-imitation-period-steps'] = int(1e9)
        CONFIG_SETS[id][0]['advice-imitation-period-samples'] = CONFIG_SETS[id][0]['advice-collection-budget']
        CONFIG_SETS[id][0]['advice-imitation-training-iterations-init'] = int(200e3)
        CONFIG_SETS[id][0]['advice-reuse-method'] = 'extended'
        CONFIG_SETS[id][0]['advice-reuse-probability'] = 0.5
        CONFIG_SETS[id][1]['autoset-teacher-model-uc-th'] = True
        CONFIG_SETS[id][1]['advice-reuse-probability-decay'] = True
        CONFIG_SETS[id][0]['advice-reuse-probability-decay-begin'] = int(500e3)
        CONFIG_SETS[id][0]['advice-reuse-probability-decay-end'] = int(2000e3)
        CONFIG_SETS[id][0]['advice-reuse-probability-final'] = 0.1

# ----------------------------------------------------------------------------------------------------------------------
# AIR: AR+A+E is enhanced with the uncertainty-based advice collection

for i, teacher_level_option in enumerate(TEACHER_LEVEL_OPTIONS):
    for j, budget_option in enumerate(BUDGET_OPTIONS):
        id = int('330' + str(i) + str(j))
        CONFIG_SETS[id] = generate_config()
        CONFIG_SETS[id][1]['load-teacher'] = True
        CONFIG_SETS[id][0]['teacher-level'] = teacher_level_option
        CONFIG_SETS[id][0]['advice-collection-budget'] = budget_option
        CONFIG_SETS[id][0]['advice-collection-method'] = 'teacher_model_uc'
        CONFIG_SETS[id][0]['advice-imitation-method'] = 'periodic'
        CONFIG_SETS[id][0]['advice-imitation-period-steps'] = int(CONFIG_SETS[id][0]['advice-collection-budget'] * 2)
        CONFIG_SETS[id][0]['advice-imitation-period-samples'] = int(CONFIG_SETS[id][0]['advice-collection-budget'] // 10)
        CONFIG_SETS[id][0]['advice-imitation-training-iterations-init'] = int(1e3)  # original: int(200e3)
        CONFIG_SETS[id][0]['advice-imitation-training-iterations-periodic'] = int(500)  # original: int(50e3)
        CONFIG_SETS[id][0]['advice-reuse-method'] = 'extended'
        CONFIG_SETS[id][0]['advice-reuse-probability'] = 0.5
        CONFIG_SETS[id][1]['autoset-teacher-model-uc-th'] = True
        CONFIG_SETS[id][1]['advice-reuse-probability-decay'] = True
        CONFIG_SETS[id][0]['advice-reuse-probability-decay-begin'] = int(2e3)
        CONFIG_SETS[id][0]['advice-reuse-probability-decay-end'] = int(5e3)
        CONFIG_SETS[id][0]['advice-reuse-probability-final'] = 0.1

        CONFIG_SETS[id][1]['dqn-twin'] = True
        CONFIG_SETS[id][0]['dqn-twin-uncertainty-type'] = 0  # Q-values variances

# AIR + Reuse Stopping
for i, teacher_level_option in enumerate(TEACHER_LEVEL_OPTIONS):
    for j, budget_option in enumerate(BUDGET_OPTIONS):
        id = int('331' + str(i) + str(j))
        CONFIG_SETS[id] = generate_config()
        CONFIG_SETS[id][1]['load-teacher'] = True
        CONFIG_SETS[id][0]['teacher-level'] = teacher_level_option
        CONFIG_SETS[id][0]['advice-collection-budget'] = budget_option
        CONFIG_SETS[id][0]['advice-collection-method'] = 'teacher_model_uc'
        CONFIG_SETS[id][0]['advice-imitation-method'] = 'periodic'
        CONFIG_SETS[id][0]['advice-imitation-period-steps'] = int(CONFIG_SETS[id][0]['advice-collection-budget'] * 2)
        CONFIG_SETS[id][0]['advice-imitation-period-samples'] = int(CONFIG_SETS[id][0]['advice-collection-budget'] // 10)
        CONFIG_SETS[id][0]['advice-imitation-training-iterations-init'] = int(50e3)  # original: int(200e3)
        CONFIG_SETS[id][0]['advice-imitation-training-iterations-periodic'] = int(20e3)  # original: int(50e3)
        CONFIG_SETS[id][0]['advice-reuse-method'] = 'extended'
        CONFIG_SETS[id][0]['advice-reuse-probability'] = 0.5
        CONFIG_SETS[id][1]['autoset-teacher-model-uc-th'] = True
        CONFIG_SETS[id][1]['advice-reuse-probability-decay'] = True
        CONFIG_SETS[id][0]['advice-reuse-probability-decay-begin'] = int(500e3)
        CONFIG_SETS[id][0]['advice-reuse-probability-decay-end'] = int(2000e3)
        CONFIG_SETS[id][0]['advice-reuse-probability-final'] = 0.1
        CONFIG_SETS[id][1]['advice-reuse-stopping'] = True
        CONFIG_SETS[id][0]['advice-reuse-stopping-eval-start'] = 50
        CONFIG_SETS[id][0]['advice-reuse-stopping-eval-window-size'] = 5
        CONFIG_SETS[id][0]['advice-reuse-stopping-eval-proximity'] = 0.9

# ----------------------------------------------------------------------------------------------------------------------
# Student model uncertainty driven collection (with constant threshold)

for i, teacher_level_option in enumerate(TEACHER_LEVEL_OPTIONS):
    for j, budget_option in enumerate(BUDGET_OPTIONS):
        id = int('400' + str(i) + str(j))
        CONFIG_SETS[id] = generate_config()
        CONFIG_SETS[id][1]['load-teacher'] = True
        CONFIG_SETS[id][0]['teacher-level'] = teacher_level_option
        CONFIG_SETS[id][0]['advice-collection-budget'] = budget_option
        CONFIG_SETS[id][0]['advice-collection-method'] = 'student_model_uc'
        CONFIG_SETS[id][1]['dqn-twin'] = True
        CONFIG_SETS[id][0]['student-model-uc-th'] = 0.01

# ----------------------------------------------------------------------------------------------------------------------
# Student model uncertainty driven collection (with adaptive threshold)

for i, teacher_level_option in enumerate(TEACHER_LEVEL_OPTIONS):
    for j, budget_option in enumerate(BUDGET_OPTIONS):
        id = int('410' + str(i) + str(j))
        CONFIG_SETS[id] = generate_config()
        CONFIG_SETS[id][1]['load-teacher'] = True
        CONFIG_SETS[id][0]['teacher-level'] = teacher_level_option
        CONFIG_SETS[id][0]['advice-collection-budget'] = budget_option
        CONFIG_SETS[id][0]['advice-collection-method'] = 'student_model_uc'
        CONFIG_SETS[id][1]['dqn-twin'] = True
        CONFIG_SETS[id][1]['use-proportional-student-model-uc-th'] = True
        CONFIG_SETS[id][0]['proportional-student-model-uc-th-window-size'] = 10000
        CONFIG_SETS[id][0]['proportional-student-model-uc-th-window-size-min'] = 200
        CONFIG_SETS[id][0]['proportional-student-model-uc-th-percentile'] = 70

# ----------------------------------------------------------------------------------------------------------------------
# Student model uncertainty driven collection (with constant threshold) + Teacher Imitation & Reuse

for i, teacher_level_option in enumerate(TEACHER_LEVEL_OPTIONS):
    for j, budget_option in enumerate(BUDGET_OPTIONS):
        id = int('500' + str(i) + str(j))
        CONFIG_SETS[id] = generate_config()
        CONFIG_SETS[id][1]['load-teacher'] = True
        CONFIG_SETS[id][0]['teacher-level'] = teacher_level_option
        CONFIG_SETS[id][0]['advice-collection-budget'] = budget_option
        CONFIG_SETS[id][0]['advice-collection-method'] = 'student_model_uc'
        CONFIG_SETS[id][1]['dqn-twin'] = True
        CONFIG_SETS[id][0]['student-model-uc-th'] = 0.01
        CONFIG_SETS[id][0]['advice-imitation-method'] = 'periodic'
        CONFIG_SETS[id][0]['advice-imitation-period-steps'] = int(CONFIG_SETS[id][0]['advice-collection-budget'] * 2)
        CONFIG_SETS[id][0]['advice-imitation-period-samples'] = int(
            CONFIG_SETS[id][0]['advice-collection-budget'] // 10)
        CONFIG_SETS[id][0]['advice-imitation-training-iterations-init'] = int(50e3)
        CONFIG_SETS[id][0]['advice-imitation-training-iterations-periodic'] = int(20e3)
        CONFIG_SETS[id][0]['advice-reuse-method'] = 'extended'
        CONFIG_SETS[id][0]['advice-reuse-probability'] = 0.5
        CONFIG_SETS[id][1]['autoset-teacher-model-uc-th'] = True
        CONFIG_SETS[id][1]['advice-reuse-probability-decay'] = True
        CONFIG_SETS[id][0]['advice-reuse-probability-decay-begin'] = int(500e3)
        CONFIG_SETS[id][0]['advice-reuse-probability-decay-end'] = int(2000e3)
        CONFIG_SETS[id][0]['advice-reuse-probability-final'] = 0.1

# ----------------------------------------------------------------------------------------------------------------------
# Student model uncertainty driven collection (with adaptive threshold) + Teacher Imitation & Reuse

for i, teacher_level_option in enumerate(TEACHER_LEVEL_OPTIONS):
    for j, budget_option in enumerate(BUDGET_OPTIONS):
        id = int('510' + str(i) + str(j))
        CONFIG_SETS[id] = generate_config()
        CONFIG_SETS[id][1]['load-teacher'] = True
        CONFIG_SETS[id][0]['teacher-level'] = teacher_level_option
        CONFIG_SETS[id][0]['advice-collection-budget'] = budget_option
        CONFIG_SETS[id][0]['advice-collection-method'] = 'student_model_uc'
        CONFIG_SETS[id][1]['dqn-twin'] = True
        CONFIG_SETS[id][1]['use-proportional-student-model-uc-th'] = True
        CONFIG_SETS[id][0]['proportional-student-model-uc-th-window-size'] = 10000
        CONFIG_SETS[id][0]['proportional-student-model-uc-th-window-size-min'] = 200
        CONFIG_SETS[id][0]['proportional-student-model-uc-th-percentile'] = 70
        CONFIG_SETS[id][0]['advice-imitation-method'] = 'periodic'
        CONFIG_SETS[id][0]['advice-imitation-period-steps'] = int(CONFIG_SETS[id][0]['advice-collection-budget'] * 2)
        CONFIG_SETS[id][0]['advice-imitation-period-samples'] = int(
            CONFIG_SETS[id][0]['advice-collection-budget'] // 10)
        CONFIG_SETS[id][0]['advice-imitation-training-iterations-init'] = int(50e3)
        CONFIG_SETS[id][0]['advice-imitation-training-iterations-periodic'] = int(20e3)
        CONFIG_SETS[id][0]['advice-reuse-method'] = 'extended'
        CONFIG_SETS[id][0]['advice-reuse-probability'] = 0.5
        CONFIG_SETS[id][1]['autoset-teacher-model-uc-th'] = True
        CONFIG_SETS[id][1]['advice-reuse-probability-decay'] = True
        CONFIG_SETS[id][0]['advice-reuse-probability-decay-begin'] = int(500e3)
        CONFIG_SETS[id][0]['advice-reuse-probability-decay-end'] = int(2000e3)
        CONFIG_SETS[id][0]['advice-reuse-probability-final'] = 0.1

# ----------------------------------------------------------------------------------------------------------------------
# Student model uncertainty driven collection (with adaptive threshold) + Teacher Imitation & Reuse + Reuse Stopping

for i, teacher_level_option in enumerate(TEACHER_LEVEL_OPTIONS):
    for j, budget_option in enumerate(BUDGET_OPTIONS):
        id = int('511' + str(i) + str(j))
        CONFIG_SETS[id] = generate_config()
        CONFIG_SETS[id][1]['load-teacher'] = True
        CONFIG_SETS[id][0]['teacher-level'] = teacher_level_option
        CONFIG_SETS[id][0]['advice-collection-budget'] = budget_option
        CONFIG_SETS[id][0]['advice-collection-method'] = 'student_model_uc'
        CONFIG_SETS[id][1]['dqn-twin'] = True
        CONFIG_SETS[id][1]['use-proportional-student-model-uc-th'] = True
        CONFIG_SETS[id][0]['proportional-student-model-uc-th-window-size'] = 10000
        CONFIG_SETS[id][0]['proportional-student-model-uc-th-window-size-min'] = 200
        CONFIG_SETS[id][0]['proportional-student-model-uc-th-percentile'] = 70
        CONFIG_SETS[id][0]['advice-imitation-method'] = 'periodic'
        CONFIG_SETS[id][0]['advice-imitation-period-steps'] = int(CONFIG_SETS[id][0]['advice-collection-budget'] * 2)
        CONFIG_SETS[id][0]['advice-imitation-period-samples'] = int(
            CONFIG_SETS[id][0]['advice-collection-budget'] // 10)
        CONFIG_SETS[id][0]['advice-imitation-training-iterations-init'] = int(50e3)
        CONFIG_SETS[id][0]['advice-imitation-training-iterations-periodic'] = int(20e3)
        CONFIG_SETS[id][0]['advice-reuse-method'] = 'extended'
        CONFIG_SETS[id][0]['advice-reuse-probability'] = 0.5
        CONFIG_SETS[id][1]['autoset-teacher-model-uc-th'] = True
        CONFIG_SETS[id][1]['advice-reuse-probability-decay'] = True
        CONFIG_SETS[id][0]['advice-reuse-probability-decay-begin'] = int(500e3)
        CONFIG_SETS[id][0]['advice-reuse-probability-decay-end'] = int(2000e3)
        CONFIG_SETS[id][0]['advice-reuse-probability-final'] = 0.1
        CONFIG_SETS[id][1]['advice-reuse-stopping'] = True
        CONFIG_SETS[id][0]['advice-reuse-stopping-eval-start'] = 50
        CONFIG_SETS[id][0]['advice-reuse-stopping-eval-window-size'] = 5
        CONFIG_SETS[id][0]['advice-reuse-stopping-eval-proximity'] = 0.9

# ----------------------------------------------------------------------------------------------------------------------
# Student model uncertainty driven collection (with adaptive threshold) + Teacher Imitation & Reuse
# with Cheating (Mistake Correction)

for i, teacher_level_option in enumerate(TEACHER_LEVEL_OPTIONS):
    for j, budget_option in enumerate(BUDGET_OPTIONS):
        id = int('530' + str(i) + str(j))
        CONFIG_SETS[id] = generate_config()
        CONFIG_SETS[id][1]['load-teacher'] = True
        CONFIG_SETS[id][0]['teacher-level'] = teacher_level_option
        CONFIG_SETS[id][0]['advice-collection-budget'] = budget_option
        CONFIG_SETS[id][0]['advice-collection-method'] = 'student_model_uc'
        CONFIG_SETS[id][1]['dqn-twin'] = True
        CONFIG_SETS[id][1]['use-proportional-student-model-uc-th'] = True
        CONFIG_SETS[id][0]['proportional-student-model-uc-th-window-size'] = 10000
        CONFIG_SETS[id][0]['proportional-student-model-uc-th-window-size-min'] = 200
        CONFIG_SETS[id][0]['proportional-student-model-uc-th-percentile'] = 70
        CONFIG_SETS[id][0]['advice-imitation-method'] = 'periodic'
        CONFIG_SETS[id][0]['advice-imitation-period-steps'] = int(CONFIG_SETS[id][0]['advice-collection-budget'] * 2)
        CONFIG_SETS[id][0]['advice-imitation-period-samples'] = int(
            CONFIG_SETS[id][0]['advice-collection-budget'] // 10)
        CONFIG_SETS[id][0]['advice-imitation-training-iterations-init'] = int(50e3)
        CONFIG_SETS[id][0]['advice-imitation-training-iterations-periodic'] = int(20e3)
        CONFIG_SETS[id][0]['advice-reuse-method'] = 'extended'
        CONFIG_SETS[id][0]['advice-reuse-probability'] = 0.5
        CONFIG_SETS[id][1]['autoset-teacher-model-uc-th'] = True
        CONFIG_SETS[id][1]['advice-reuse-probability-decay'] = True
        CONFIG_SETS[id][0]['advice-reuse-probability-decay-begin'] = int(500e3)
        CONFIG_SETS[id][0]['advice-reuse-probability-decay-end'] = int(2000e3)
        CONFIG_SETS[id][0]['advice-reuse-probability-final'] = 0.1
        CONFIG_SETS[id][1]['mistake-correction-mode'] = True

# ----------------------------------------------------------------------------------------------------------------------
# Student model uncertainty driven collection (with adaptive threshold) + Teacher Imitation & Reuse + Reuse Stopping
# with Cheating (Mistake Correction)

for i, teacher_level_option in enumerate(TEACHER_LEVEL_OPTIONS):
    for j, budget_option in enumerate(BUDGET_OPTIONS):
        id = int('531' + str(i) + str(j))
        CONFIG_SETS[id] = generate_config()
        CONFIG_SETS[id][1]['load-teacher'] = True
        CONFIG_SETS[id][0]['teacher-level'] = teacher_level_option
        CONFIG_SETS[id][0]['advice-collection-budget'] = budget_option
        CONFIG_SETS[id][0]['advice-collection-method'] = 'student_model_uc'
        CONFIG_SETS[id][1]['dqn-twin'] = True
        CONFIG_SETS[id][1]['use-proportional-student-model-uc-th'] = True
        CONFIG_SETS[id][0]['proportional-student-model-uc-th-window-size'] = 10000
        CONFIG_SETS[id][0]['proportional-student-model-uc-th-window-size-min'] = 200
        CONFIG_SETS[id][0]['proportional-student-model-uc-th-percentile'] = 70
        CONFIG_SETS[id][0]['advice-imitation-method'] = 'periodic'
        CONFIG_SETS[id][0]['advice-imitation-period-steps'] = int(CONFIG_SETS[id][0]['advice-collection-budget'] * 2)
        CONFIG_SETS[id][0]['advice-imitation-period-samples'] = int(
            CONFIG_SETS[id][0]['advice-collection-budget'] // 10)
        CONFIG_SETS[id][0]['advice-imitation-training-iterations-init'] = int(50e3)
        CONFIG_SETS[id][0]['advice-imitation-training-iterations-periodic'] = int(20e3)
        CONFIG_SETS[id][0]['advice-reuse-method'] = 'extended'
        CONFIG_SETS[id][0]['advice-reuse-probability'] = 0.5
        CONFIG_SETS[id][1]['autoset-teacher-model-uc-th'] = True
        CONFIG_SETS[id][1]['advice-reuse-probability-decay'] = True
        CONFIG_SETS[id][0]['advice-reuse-probability-decay-begin'] = int(500e3)
        CONFIG_SETS[id][0]['advice-reuse-probability-decay-end'] = int(2000e3)
        CONFIG_SETS[id][0]['advice-reuse-probability-final'] = 0.1
        CONFIG_SETS[id][1]['advice-reuse-stopping'] = True
        CONFIG_SETS[id][0]['advice-reuse-stopping-eval-start'] = 50
        CONFIG_SETS[id][0]['advice-reuse-stopping-eval-window-size'] = 5
        CONFIG_SETS[id][0]['advice-reuse-stopping-eval-proximity'] = 0.9
        CONFIG_SETS[id][1]['mistake-correction-mode'] = True

# ----------------------------------------------------------------------------------------------------------------------
# Dual uncertainty (with constant threshold for the student model)

for i, teacher_level_option in enumerate(TEACHER_LEVEL_OPTIONS):
    for j, budget_option in enumerate(BUDGET_OPTIONS):
        id = int('600' + str(i) + str(j))
        CONFIG_SETS[id] = generate_config()
        CONFIG_SETS[id][1]['load-teacher'] = True
        CONFIG_SETS[id][0]['teacher-level'] = teacher_level_option
        CONFIG_SETS[id][0]['advice-collection-budget'] = budget_option
        CONFIG_SETS[id][0]['advice-collection-method'] = 'dual_uc'
        CONFIG_SETS[id][1]['dqn-twin'] = True
        CONFIG_SETS[id][0]['student-model-uc-th'] = 0.01
        CONFIG_SETS[id][0]['advice-imitation-method'] = 'periodic'
        CONFIG_SETS[id][0]['advice-imitation-period-steps'] = int(CONFIG_SETS[id][0]['advice-collection-budget'] * 2)
        CONFIG_SETS[id][0]['advice-imitation-period-samples'] = int(
            CONFIG_SETS[id][0]['advice-collection-budget'] // 10)
        CONFIG_SETS[id][0]['advice-imitation-training-iterations-init'] = int(50e3)
        CONFIG_SETS[id][0]['advice-imitation-training-iterations-periodic'] = int(20e3)
        CONFIG_SETS[id][1][
            'autoset-teacher-model-uc-th'] = True  # A constant threshold can also be used instead of auto setting
        CONFIG_SETS[id][0]['advice-reuse-probability'] = 1.0
        CONFIG_SETS[id][1]['advice-reuse-probability-decay'] = False

# ----------------------------------------------------------------------------------------------------------------------
# Dual uncertainty (with adaptive threshold for the student model) with 1.0 probability of reuse (when appropriate)

for i, teacher_level_option in enumerate(TEACHER_LEVEL_OPTIONS):
    for j, budget_option in enumerate(BUDGET_OPTIONS):
        id = int('610' + str(i) + str(j))
        CONFIG_SETS[id] = generate_config()
        CONFIG_SETS[id][1]['load-teacher'] = True
        CONFIG_SETS[id][0]['teacher-level'] = teacher_level_option
        CONFIG_SETS[id][0]['advice-collection-budget'] = budget_option
        CONFIG_SETS[id][0]['advice-collection-method'] = 'dual_uc'
        CONFIG_SETS[id][1]['dqn-twin'] = True
        CONFIG_SETS[id][1]['use-proportional-student-model-uc-th'] = True
        CONFIG_SETS[id][0]['proportional-student-model-uc-th-window-size'] = 10000
        CONFIG_SETS[id][0]['proportional-student-model-uc-th-window-size-min'] = 200
        CONFIG_SETS[id][0]['proportional-student-model-uc-th-percentile'] = 70
        CONFIG_SETS[id][0]['advice-imitation-method'] = 'periodic'
        CONFIG_SETS[id][0]['advice-imitation-period-steps'] = int(CONFIG_SETS[id][0]['advice-collection-budget'] * 2)
        CONFIG_SETS[id][0]['advice-imitation-period-samples'] = int(
            CONFIG_SETS[id][0]['advice-collection-budget'] // 10)
        CONFIG_SETS[id][0]['advice-imitation-training-iterations-init'] = int(50e3)
        CONFIG_SETS[id][0]['advice-imitation-training-iterations-periodic'] = int(20e3)
        CONFIG_SETS[id][1][
            'autoset-teacher-model-uc-th'] = True  # A constant threshold can also be used instead of auto setting
        CONFIG_SETS[id][0]['advice-reuse-probability'] = 1.0
        CONFIG_SETS[id][1]['advice-reuse-probability-decay'] = False

# ----------------------------------------------------------------------------------------------------------------------
# Dual uncertainty (with adaptive threshold for the student model) with 0.5 probability of reuse (when appropriate)

for i, teacher_level_option in enumerate(TEACHER_LEVEL_OPTIONS):
    for j, budget_option in enumerate(BUDGET_OPTIONS):
        id = int('611' + str(i) + str(j))
        CONFIG_SETS[id] = generate_config()
        CONFIG_SETS[id][1]['load-teacher'] = True
        CONFIG_SETS[id][0]['teacher-level'] = teacher_level_option
        CONFIG_SETS[id][0]['advice-collection-budget'] = budget_option
        CONFIG_SETS[id][0]['advice-collection-method'] = 'dual_uc'
        CONFIG_SETS[id][1]['dqn-twin'] = True
        CONFIG_SETS[id][1]['use-proportional-student-model-uc-th'] = True
        CONFIG_SETS[id][0]['proportional-student-model-uc-th-window-size'] = 10000
        CONFIG_SETS[id][0]['proportional-student-model-uc-th-window-size-min'] = 200
        CONFIG_SETS[id][0]['proportional-student-model-uc-th-percentile'] = 70
        CONFIG_SETS[id][0]['advice-imitation-method'] = 'periodic'
        CONFIG_SETS[id][0]['advice-imitation-period-steps'] = int(CONFIG_SETS[id][0]['advice-collection-budget'] * 2)
        CONFIG_SETS[id][0]['advice-imitation-period-samples'] = int(
            CONFIG_SETS[id][0]['advice-collection-budget'] // 10)
        CONFIG_SETS[id][0]['advice-imitation-training-iterations-init'] = int(50e3)
        CONFIG_SETS[id][0]['advice-imitation-training-iterations-periodic'] = int(20e3)
        CONFIG_SETS[id][1][
            'autoset-teacher-model-uc-th'] = True  # A constant threshold can also be used instead of auto setting
        CONFIG_SETS[id][0]['advice-reuse-probability'] = 0.5
        CONFIG_SETS[id][1]['advice-reuse-probability-decay'] = False

# ----------------------------------------------------------------------------------------------------------------------
# Dual uncertainty (with adaptive threshold for the student model) with decaying probability of reuse (when appropriate)

for i, teacher_level_option in enumerate(TEACHER_LEVEL_OPTIONS):
    for j, budget_option in enumerate(BUDGET_OPTIONS):
        id = int('612' + str(i) + str(j))
        CONFIG_SETS[id] = generate_config()
        CONFIG_SETS[id][1]['load-teacher'] = True
        CONFIG_SETS[id][0]['teacher-level'] = teacher_level_option
        CONFIG_SETS[id][0]['advice-collection-budget'] = budget_option
        CONFIG_SETS[id][0]['advice-collection-method'] = 'dual_uc'
        CONFIG_SETS[id][1]['dqn-twin'] = True
        CONFIG_SETS[id][1]['use-proportional-student-model-uc-th'] = True
        CONFIG_SETS[id][0]['proportional-student-model-uc-th-window-size'] = 10000
        CONFIG_SETS[id][0]['proportional-student-model-uc-th-window-size-min'] = 200
        CONFIG_SETS[id][0]['proportional-student-model-uc-th-percentile'] = 70
        CONFIG_SETS[id][0]['advice-imitation-method'] = 'periodic'
        CONFIG_SETS[id][0]['advice-imitation-period-steps'] = int(CONFIG_SETS[id][0]['advice-collection-budget'] * 2)
        CONFIG_SETS[id][0]['advice-imitation-period-samples'] = int(
            CONFIG_SETS[id][0]['advice-collection-budget'] // 10)
        CONFIG_SETS[id][0]['advice-imitation-training-iterations-init'] = int(50e3)
        CONFIG_SETS[id][0]['advice-imitation-training-iterations-periodic'] = int(20e3)
        CONFIG_SETS[id][1][
            'autoset-teacher-model-uc-th'] = True  # A constant threshold can also be used instead of auto setting
        CONFIG_SETS[id][0]['advice-reuse-probability'] = 1.0
        CONFIG_SETS[id][1]['advice-reuse-probability-decay'] = True
        CONFIG_SETS[id][0]['advice-reuse-probability-decay-begin'] = int(500e3)
        CONFIG_SETS[id][0]['advice-reuse-probability-decay-end'] = int(2000e3)
        CONFIG_SETS[id][0]['advice-reuse-probability-final'] = 0.1

# ----------------------------------------------------------------------------------------------------------------------

for i, teacher_level_option in enumerate(TEACHER_LEVEL_OPTIONS):
    for j, budget_option in enumerate(BUDGET_OPTIONS):
        id = int('700' + str(i) + str(j))
        CONFIG_SETS[id] = generate_config()
        CONFIG_SETS[id][1]['load-teacher'] = True
        CONFIG_SETS[id][0]['teacher-level'] = teacher_level_option
        CONFIG_SETS[id][0]['advice-collection-budget'] = budget_option
        CONFIG_SETS[id][0]['advice-collection-method'] = 'dual_uc'
        CONFIG_SETS[id][1]['dqn-twin'] = True
        CONFIG_SETS[id][1]['use-proportional-student-model-uc-th'] = True
        CONFIG_SETS[id][0]['proportional-student-model-uc-th-window-size'] = 10000
        CONFIG_SETS[id][0]['proportional-student-model-uc-th-window-size-min'] = 200
        CONFIG_SETS[id][0]['proportional-student-model-uc-th-percentile'] = 70
        CONFIG_SETS[id][0]['advice-imitation-method'] = 'periodic'
        CONFIG_SETS[id][0]['advice-imitation-period-steps'] = int(CONFIG_SETS[id][0]['advice-collection-budget'] * 2)
        CONFIG_SETS[id][0]['advice-imitation-period-samples'] = int(
            CONFIG_SETS[id][0]['advice-collection-budget'] // 10)
        CONFIG_SETS[id][0]['advice-imitation-training-iterations-init'] = int(50e3)
        CONFIG_SETS[id][0]['advice-imitation-training-iterations-periodic'] = int(20e3)
        CONFIG_SETS[id][1][
            'autoset-teacher-model-uc-th'] = True  # A constant threshold can also be used instead of auto setting
        CONFIG_SETS[id][0]['advice-reuse-probability'] = 1.0
        CONFIG_SETS[id][0]['advice-reuse-probability-final'] = 1.0
        CONFIG_SETS[id][1]['advice-reuse-probability-decay'] = False
        CONFIG_SETS[id][1]['advice-reuse-stopping'] = True
        CONFIG_SETS[id][0]['advice-reuse-stopping-eval-start'] = 50
        CONFIG_SETS[id][0]['advice-reuse-stopping-eval-window-size'] = 5
        CONFIG_SETS[id][0]['advice-reuse-stopping-eval-proximity'] = 0.9

# ----------------------------------------------------------------------------------------------------------------------


