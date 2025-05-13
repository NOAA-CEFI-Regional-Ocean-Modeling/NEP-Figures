library(coldpool)
library(akgfmaps)

proj_crs <- coldpool:::ebs_proj_crs
experiment <- "MBC_FIX"
experiment_dir <- paste('/work/Utheri.Wagura/NEP/plotting/Figure_19_20',experiment, sep='/')
ebs_nep_csv_path <- paste( experiment_dir, 'index_hauls_temperature_data_nep.csv', sep='/')
print( paste("Working with the following ebs file:",ebs_nep_csv_path, sep = " " ) )

# Interpolate gear temperature and write rasters for SEBS - nep bottom temp values
print("Interpolating index hauls")
interpolation_wrapper(temp_data_path = ebs_nep_csv_path,
                      proj_crs = proj_crs,
                      cell_resolution = 5000, # 5x5 km grid resolution
                      select_years = 1993:2024,# NOTE: This should be 2024
                      interp_variable = "nep_tob",
                      select_region = "sebs",
                      methods = "Ste")

# Calculate cold pool area for NEP
print("Calculate Cold Pool Area for NEP")
bottom_temp_files <- list.files(here::here("output", "raster", "sebs", "nep_tob"),
                                full.names = TRUE,
                                pattern = "ste_")

print("Writing dataframe")
bt_df <- data.frame(YEAR = numeric(length = length(bottom_temp_files)),
                    AREA_LTE2_KM2 = numeric(length = length(bottom_temp_files)),
                    AREA_LTE1_KM2 = numeric(length = length(bottom_temp_files)),
                    AREA_LTE0_KM2 = numeric(length = length(bottom_temp_files)),
                    AREA_LTEMINUS1_KM2 = numeric(length = length(bottom_temp_files)),
                    MEAN_GEAR_TEMPERATURE = numeric(length = length(bottom_temp_files)),
                    MEAN_BT_LT100M = numeric(length = length(bottom_temp_files)))

# Setup mask to calculate mean bottom temperature from <100 m strata
print("Setting up mask to calculate mean bottom temp")
ebs_layers <- akgfmaps::get_base_layers(select.region = "sebs", set.crs = proj_crs)

print("Creating lt100 strata")
lt100_strata <- ebs_layers$survey.strata |>
  #dplyr::filter(STRATUM %in% c(10, 20, 31, 32, 41, 42, 43)) |> #NOTE: Use these keys in v4
  #dplyr::group_by(SURVEY_DEFINITION_ID) |> #NOTE: THIS IS NOT THE SAME THING AS SURVEY, SEE JOB_OUTPUT_FILES/FIX and work on debugging this!
  dplyr::filter(Stratum %in% c(10, 20, 31, 32, 41, 42, 43)) |>
  dplyr::group_by(SURVEY) |>
  dplyr::summarise()


print("Looping over bottom temp files")
for(i in 1:length(bottom_temp_files)) {
  bt_raster <- terra::rast(bottom_temp_files[i])
  bt_df$YEAR[i] <- as.numeric(gsub("[^0-9.-]", "", names(bt_raster))) # Extract year
  bt_df$AREA_LTE2_KM2[i] <- bt_raster |>
    cpa_from_raster(raster_units = "m", temperature_threshold = 2)
  bt_df$AREA_LTE1_KM2[i] <- bt_raster |>
    cpa_from_raster(raster_units = "m", temperature_threshold = 1)
  bt_df$AREA_LTE0_KM2[i] <- bt_raster |>
    cpa_from_raster(raster_units = "m", temperature_threshold = 0)
  bt_df$AREA_LTEMINUS1_KM2[i] <- bt_raster |>
    cpa_from_raster(raster_units = "m", temperature_threshold = -1)
  bt_df$MEAN_GEAR_TEMPERATURE[i] <- mean(terra::values(bt_raster), na.rm = TRUE)
  lt100_temp <- terra::mask(bt_raster,
                            lt100_strata,
                            touches = FALSE)
  bt_df$MEAN_BT_LT100M[i] <- mean(terra::values(lt100_temp), na.rm = TRUE)

}

print("Writing nep_cpa.csv")
write.csv(bt_df, paste( experiment_dir, "nep_cpa.csv", sep="/") , row.names=FALSE)

# Interpolate gear temperature and write rasters for SEBS
interpolation_wrapper(temp_data_path = ebs_nep_csv_path,
                      proj_crs = proj_crs,
                      cell_resolution = 5000, # 5x5 km grid resolution
                      select_years = c(1993:2019,2021:2023), # NOTE: 2020 data is missing, and this should go to 2024
                      interp_variable = "gear_temperature",
                      select_region = "sebs",
                      methods = "Ste")

# Calculate cold pool area
print("Saving bottom temp files as var")
bottom_temp_files <- list.files(here::here("output", "raster", "sebs", "gear_temperature"),
                                full.names = TRUE,
                                pattern = "ste_")

print("Writing Data frame")
bt_df <- data.frame(YEAR = numeric(length = length(bottom_temp_files)),
                    AREA_LTE2_KM2 = numeric(length = length(bottom_temp_files)),
                    AREA_LTE1_KM2 = numeric(length = length(bottom_temp_files)),
                    AREA_LTE0_KM2 = numeric(length = length(bottom_temp_files)),
                    AREA_LTEMINUS1_KM2 = numeric(length = length(bottom_temp_files)),
                    MEAN_GEAR_TEMPERATURE = numeric(length = length(bottom_temp_files)),
                    MEAN_BT_LT100M = numeric(length = length(bottom_temp_files)))

# Setup mask to calculate mean bottom temperature from <100 m strata
print("Setting up mask to calculate mean bottom temp")
ebs_layers <- akgfmaps::get_base_layers(select.region = "sebs", set.crs = proj_crs)

print("lt100_strat")
lt100_strata <- ebs_layers$survey.strata |>
  #dplyr::filter(STRATUM %in% c(10, 20, 31, 32, 41, 42, 43)) |>
  #dplyr::group_by(SURVEY_DEFINITION_ID) |>
  dplyr::filter(Stratum %in% c(10, 20, 31, 32, 41, 42, 43)) |>
  dplyr::group_by(SURVEY) |>
  dplyr::summarise()

print("Listing bottom temp files")
for(i in 1:length(bottom_temp_files)) {
  bt_raster <- terra::rast(bottom_temp_files[i])
  bt_df$YEAR[i] <- as.numeric(gsub("[^0-9.-]", "", names(bt_raster))) # Extract year
  bt_df$AREA_LTE2_KM2[i] <- bt_raster |>
    cpa_from_raster(raster_units = "m", temperature_threshold = 2)
  bt_df$AREA_LTE1_KM2[i] <- bt_raster |>
    cpa_from_raster(raster_units = "m", temperature_threshold = 1)
  bt_df$AREA_LTE0_KM2[i] <- bt_raster |>
    cpa_from_raster(raster_units = "m", temperature_threshold = 0)
  bt_df$AREA_LTEMINUS1_KM2[i] <- bt_raster |>
    cpa_from_raster(raster_units = "m", temperature_threshold = -1)
  bt_df$MEAN_GEAR_TEMPERATURE[i] <- mean(terra::values(bt_raster), na.rm = TRUE)
  lt100_temp <- terra::mask(bt_raster,
                            lt100_strata,
                            touches = FALSE)
  bt_df$MEAN_BT_LT100M[i] <- mean(terra::values(lt100_temp), na.rm = TRUE)

}

print("Writing trawl csv")
write.csv(bt_df, paste(experiment_dir, "trawl_cpa.csv", sep = "/") , row.names=FALSE)
