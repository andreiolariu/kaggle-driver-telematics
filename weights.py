import model_def
import model_run

### - not run on train
#-- - ran on train, small influence
#== - ran on train, not run on test because slow
MODELS = [
  #(52, model_run.get_data_accel, model_def.Model_ABC1, 1),
  #(53, model_run.get_data_accel, model_def.Model_ABC2, 1),
  #(1, model_run.get_data_accel, model_def.Model_ETC, 1),
  (2, model_run.get_data_accel, model_def.Model_GBC, 10),
  (3, model_run.get_data_accel, model_def.Model_LR, 10),
  (63, model_run.get_data_accel, model_def.Model_LR2, 10),
  #--(4, model_run.get_data_accel, model_def.Model_RFC, 10),
  (5, model_run.get_data_accel, model_def.Model_SVC, 10), #maybe add

  # (59, model_run.get_data_accel, model_def.Model_GBC, 100),

  (103, model_run.get_data_basic_big, model_def.Model_GBC, 10),
  (101, model_run.get_data_basic_big, model_def.Model_GBC2, 10),
  (104, model_run.get_data_basic_big, model_def.Model_GBC3, 10),

  #(57, model_run.get_data_basic, model_def.Model_ABC1, 1),
  #(6, model_run.get_data_basic, model_def.Model_ETC, 1),
  (7, model_run.get_data_basic, model_def.Model_GBC, 10),
  (8, model_run.get_data_basic, model_def.Model_LR, 10),
  (64, model_run.get_data_basic, model_def.Model_LR2, 10),
  #(9, model_run.get_data_basic, model_def.Model_RFC, 1),
  #(10, model_run.get_data_basic, model_def.Model_SVC, 1), # useless # RuntimeWarning: overflow encountered in exp

  #(11, model_run.get_data_basic_accel, model_def.Model_ETC, 1),
  (12, model_run.get_data_basic_accel, model_def.Model_GBC, 10),
  #--(13, model_run.get_data_basic_accel, model_def.Model_LR, 10), # maybe add # Warning: overflow encountered in exp
  #(14, model_run.get_data_basic_accel, model_def.Model_RFC, 1),
  #(15, model_run.get_data_basic_accel, model_def.Model_SVC, 1),

  (91, model_run.get_data_basic_accel_v2, model_def.Model_GBC, 10),

  #(61, model_run.get_data_accel_v2, model_def.Model_GBC, 10),
  (72, model_run.get_data_accel_v2_svd, model_def.Model_GBC, 10),

  #(58, model_run.get_data_basic_v2, model_def.Model_ABC1, 1),
  (16, model_run.get_data_basic_v2, model_def.Model_GBC, 10),
  #(17, model_run.get_data_basic_v2, model_def.Model_LR, 1),
  #(65, model_run.get_data_basic_v2, model_def.Model_LR3, 10),
  #(18, model_run.get_data_basic_v2, model_def.Model_RFC, 1),

  (87, model_run.get_data_basic_v3, model_def.Model_GBC, 10),
  #--? (88, model_run.get_data_basic_v3, model_def.Model_LR, 10),
  (89, model_run.get_data_basic_v4, model_def.Model_GBC, 10),
  (102, model_run.get_data_basic_v4, model_def.Model_GBC2, 10),
  (105, model_run.get_data_basic_v4, model_def.Model_GBC3, 10),
  #--? (90, model_run.get_data_basic_v4, model_def.Model_LR, 10),
  (99, model_run.get_data_basic_v5, model_def.Model_GBC, 10),

  (107, model_run.get_data_dist_acc, model_def.Model_LR, 4),
  (108, model_run.get_data_dist_acc, model_def.Model_LR2, 4),
  (109, model_run.get_data_dist_acc, model_def.Model_SVC, 4),

  (76, model_run.get_data_fft, model_def.Model_ETC, 10),
  (77, model_run.get_data_fft_v2, model_def.Model_LR3, 10),
  (78, model_run.get_data_fft_v2, model_def.Model_SVC2, 10),

  #(19, model_run.get_data_g_forces_v1, model_def.Model_LR, 1), # slow, maybe add
  (94, model_run.get_data_g_forces_v1, model_def.Model_LR2, 1), # slow, maybe add
  (95, model_run.get_data_g_forces_v1, model_def.Model_SVC, 1), # slow, maybe add
  ## (20, model_run.get_data_g_forces_v1, model_def.Model_SVC, 1), maybe add
  # (21, model_run.get_data_g_forces_v2, model_def.Model_SVC, 4),
  #(22, model_run.get_data_g_forces_v3, model_def.Model_RFC, 1),
  (62, model_run.get_data_g_forces_v4, model_def.Model_SVC, 4),
  (93, model_run.get_data_g_forces_v5, model_def.Model_LR, 10),
  (96, model_run.get_data_g_forces_v6, model_def.Model_LR2, 1),
  (97, model_run.get_data_g_forces_v6, model_def.Model_SVC, 1),
  (98, model_run.get_data_g_forces_v7, model_def.Model_LR2, 4),

  #(48, model_run.get_data_heading, model_def.Model_ETC, 1), # 15 minutes per user, maybe add
  #(48, model_run.get_data_heading, model_def.Model_GBC, 1), # 8 minutes per user, nerulat pe train
  (49, model_run.get_data_heading, model_def.Model_LR, 10),
  # (50, model_run.get_data_heading, model_def.Model_RFC, 1),
  (51, model_run.get_data_heading, model_def.Model_SVC, 10),
  # (54, model_run.get_data_heading, model_def.Model_SVC2, 1), # slow, not add?

  (79, model_run.get_data_heading_stops, model_def.Model_LR, 10),
  #-- (85, model_run.get_data_heading_stops_v2, model_def.Model_LR, 10),

  (55, model_run.get_data_heading_v2, model_def.Model_LR, 10),
  (56, model_run.get_data_heading_v2, model_def.Model_SVC, 10),

  (86, model_run.get_data_heading_v3, model_def.Model_LR, 10),

  # ?? (74, model_run.get_data_movements_accel, model_def.Model_LR, 4),
  # ?? (75, model_run.get_data_movements_accel, model_def.Model_LR2, 4),
  (60, model_run.get_data_movements_accel, model_def.Model_SVC, 4),
  #(73, model_run.get_data_movements_accel_svd, model_def.Model_GBC, 4),
  (92, model_run.get_data_movements_accel_v2, model_def.Model_SVC, 4),

  (66, model_run.get_data_movements_v1, model_def.Model_LR, 10),
  (69, model_run.get_data_movements_v1, model_def.Model_LR2, 10),
  #(23, model_run.get_data_movements_v1, model_def.Model_RFC, 1),
  (24, model_run.get_data_movements_v1, model_def.Model_SVC, 10),
  #(25, model_run.get_data_movements_v1_tf, model_def.Model_SVC, 1),

  #(26, model_run.get_data_movements_v2, model_def.Model_LR, 1), # folds=10, not add
  (67, model_run.get_data_movements_v2, model_def.Model_LR2, 10),
  (27, model_run.get_data_movements_v2, model_def.Model_SVC, 10), # slow
  #(28, model_run.get_data_movements_v2, model_def.Model_SVC2, 1), # folds=10, maybe add
  #(29, model_run.get_data_movements_v2_tf, model_def.Model_SVC, 1),
  (71, model_run.get_data_movements_v2_svd, model_def.Model_GBC, 10),

  #(30, model_run.get_data_movements_v3, model_def.Model_LR, 1), # folds=10, not add
  (68, model_run.get_data_movements_v3, model_def.Model_LR2, 10),
  #--(31, model_run.get_data_movements_v3, model_def.Model_SVC, 10), # maybe add # folds=10
  #(32, model_run.get_data_movements_v3, model_def.Model_SVC2, 1), # maybe add
  #(33, model_run.get_data_movements_v3_tf, model_def.Model_LR, 1),
  #(34, model_run.get_data_movements_v3_tf, model_def.Model_SVC, 1),

  #-- (81, model_run.get_data_movements_v4, model_def.Model_LR2, 10),
  #-- (82, model_run.get_data_movements_v5, model_def.Model_LR2, 10),
  (83, model_run.get_data_movements_v6, model_def.Model_LR2, 10),
  (84, model_run.get_data_movements_v7, model_def.Model_LR2, 10),
  (100, model_run.get_data_movements_v8, model_def.Model_LR2, 10),

  #(35, model_run.get_data_segment_angles, model_def.Model_GBC, 1),
  (36, model_run.get_data_segment_angles, model_def.Model_LR, 4),
  #(37, model_run.get_data_segment_angles, model_def.Model_RFC, 1),
  #(38, model_run.get_data_segment_angles, model_def.Model_SVC, 1),
  #-- (80, model_run.get_data_segment_angles_v2, model_def.Model_LR, 4),

  #(40, model_run.get_data_segment_lengths, model_def.Model_GBC, 1), # folds=8, maybe add
  (41, model_run.get_data_segment_lengths, model_def.Model_LR, 4),
  #(42, model_run.get_data_segment_lengths, model_def.Model_RFC, 1),
  #(43, model_run.get_data_segment_lengths, model_def.Model_SVC, 1),

  #(44, model_run.get_data_segment_times, model_def.Model_GBC, 1), # folds=8, maybe add
  (45, model_run.get_data_segment_times, model_def.Model_LR, 4),
  #(46, model_run.get_data_segment_times, model_def.Model_RFC, 1),
  #(47, model_run.get_data_segment_times, model_def.Model_SVC, 1),

  (70, model_run.get_data_segment_v2, model_def.Model_LR, 10),

]

# calculat 107 pe test
STACK = [
  (2, model_run.get_data_accel, model_def.Model_GBC, 10), # 1 min per driver
  (3, model_run.get_data_accel, model_def.Model_LR, 10), # 40 sec per driver
  (63, model_run.get_data_accel, model_def.Model_LR2, 10), # 10 sec per driver
  (5, model_run.get_data_accel, model_def.Model_SVC, 10), # 40 sec per driver #maybe add

  (12, model_run.get_data_basic_accel, model_def.Model_GBC, 10), # 1.5 min per driver
  (91, model_run.get_data_basic_accel_v2, model_def.Model_GBC, 10), # 1.8 min per driver

  (104, model_run.get_data_basic_big, model_def.Model_GBC3, 10),
  (111, model_run.get_data_basic_big_v2, model_def.Model_GBC3, 10), # 40 sec per driver

  (16, model_run.get_data_basic_v2, model_def.Model_GBC, 10), # 40 sec per driver
  (87, model_run.get_data_basic_v3, model_def.Model_GBC, 10), # 1 min per driver
  (89, model_run.get_data_basic_v4, model_def.Model_GBC, 10), # 1.15 min per driver

  (107, model_run.get_data_dist_acc, model_def.Model_LR, 4), # 10 folds

  (96, model_run.get_data_g_forces_v6, model_def.Model_LR2, 1), # 2.5 min per driver

  (51, model_run.get_data_heading, model_def.Model_SVC, 10), # 3.5 min per driver
  (79, model_run.get_data_heading_stops, model_def.Model_LR, 10), # 3.5 min per driver
  (55, model_run.get_data_heading_v2, model_def.Model_LR, 10), # 3.5 min per driver

  (60, model_run.get_data_movements_accel, model_def.Model_SVC, 4), # 3.5 min per driver
  (67, model_run.get_data_movements_v2, model_def.Model_LR2, 10), # 4 min per driver for 8 folds
  (83, model_run.get_data_movements_v6, model_def.Model_LR2, 10),# 1 min per driver
  (84, model_run.get_data_movements_v7, model_def.Model_LR2, 10), # 1.3 min per driver

  (36, model_run.get_data_segment_angles, model_def.Model_LR, 4), # 1 sec per driver
  (41, model_run.get_data_segment_lengths, model_def.Model_LR, 4), # 3 sec per driver
  (70, model_run.get_data_segment_v2, model_def.Model_LR, 10), # 2 sec per driver
]

WEIGHTS = {2: 0.024819412691444782, 3: 0.045810432596201198, 5: 9.8381587787062545e-05, 12: 0.085751071851815894, 16: 0.036259536220486756, 36: 0.042808218136559099, 41: 0.029705081295213608, 51: 0.0453111571766193, 55: 0.067791960096966822, 60: 0.12838024895888966, 63: 0.035268117984193298, 67: 0.076977632203712093, 70: 0.01909734053123226, 79: 0.022575077886886973, 83: 0.11404680902846333, 84: 0.11611119742557031, 87: 0.049168388058686022, 89: 0.047098612253309559, 91: 0.068358155065302337, 96: 0.077934935592705035, 104: 0.039462792122427037, 111: 0.171981317575196}
