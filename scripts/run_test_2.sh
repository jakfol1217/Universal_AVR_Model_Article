# 1st argument - checkpoing_path,
# 2nd argument - slurm_id dependency (default=1)
# 3rd argument - additional hydra config (e.g. +increment_dataloader_idx=1)
test_prepare () {
    CHECKPOINT_PATH=$1
    LAST_ID=${2:-1}
    ADDITIONAL_PARAMS=${3}
    echo "Params:"
    echo $CHECKPOINT_PATH
    echo $LAST_ID
    echo $ADDITIONAL_PARAMS
}

# arugments - any number of tasks to run
test_run () {
    echo "Slurm ids:"
    for task_nm in ${@}; do
        LAST_ID=$(sbatch --parsable --time=0-00:30:00 --dependency=afterany:${LAST_ID} scripts/run_yolo.sh src/test.py "checkpoint_path='${CHECKPOINT_PATH}'" data/tasks=[${task_nm}] ${ADDITIONAL_PARAMS})
        echo -n "${LAST_ID} "
    done
    echo
}

test_bongard_logo () {
    test_prepare $1 $2 $3

    test_run bongard_logo_test_bd_vit_2 bongard_logo_test_ff_vit_2 bongard_logo_test_hd_comb_vit_2 bongard_logo_test_hd_novel_vit_2
}

test_vaec () {
    test_prepare $1 $2 $3

    test_run vaec_test1_vit_2 vaec_test2_vit_2 vaec_test3_vit_2 vaec_test4_vit_2 vaec_test5_vit_2
}

test_bongard_hoi () {
    test_prepare $1 $2 $3

    test_run bongard_hoi_seen-seen_vit_2 bongard_hoi_seen-unseen_vit_2 bongard_hoi_unseen-seen_vit_2 bongard_hoi_unseen-unseen_vit_2
}



test_vasr () {
    test_prepare $1 $2 $3
    
    test_run vasr_vit_2
}


test_iraven () {
    test_prepare $1 $2 $3
    
    test_run iraven_vit_2
}


test_clevr () {
    test_prepare $1 $2 $3

    test_run clevr_vit_2
}

test_clevr_problem1 () {
    test_prepare $1 $2 $3

    test_run clevr_problem1_vit_2
}

test_clevr_problem2 () {
    test_prepare $1 $2 $3

    test_run clevr_problem2_vit_2
}

test_clevr_problem3 () {
    test_prepare $1 $2 $3

    test_run clevr_problem3_vit_2
}


test_iraven_center_single () {
    test_prepare $1 $2 $3
    
    test_run iraven_center_single_vit_2
}

test_iraven_in_center_single_out_center_single () {
    test_prepare $1 $2 $3
    
    test_run iraven_in_center_single_out_center_single_vit_2
}

test_iraven_up_center_single_down_center_single () {
    test_prepare $1 $2 $3
    
    test_run iraven_up_center_single_down_center_single_vit_2
}

test_iraven_distribute_four () {
    test_prepare $1 $2 $3
    
    test_run iraven_distribute_four_vit_2
}

test_iraven_in_distribute_four_out_center_single () {
    test_prepare $1 $2 $3
    
    test_run iraven_in_distribute_four_out_center_single_vit_2
}

test_iraven_distribute_nine () {
    test_prepare $1 $2 $3
    
    test_run iraven_distribute_nine_vit_2
}

test_iraven_left_center_single_right_center_single () {
    test_prepare $1 $2 $3
    
    test_run iraven_left_center_single_right_center_single_vit_2
}


test_svrt () {
    test_prepare $1 $2 $3
    
    test_run svrt_vit_2
}


test_dsprites () {
    test_prepare $1 $2 $3
    
    test_run dsprites_vit_2
}


test_labc () {
    test_prepare $1 $2 $3
    
    test_run labc_vit_2
}



