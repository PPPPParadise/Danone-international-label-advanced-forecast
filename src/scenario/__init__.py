# coding: utf-8
F_CONNECT = 'CONNECT'
F_DATABASE = 'DATABASE'
F_RESULTS = 'RESULTS'
F_TABLE = 'TABLE'
F_FEATURE_IMPORTANCE = 'FEATURE_IMPORTANCE'

V_AF = 'AF'
V_EIB = 'EIB'
V_DI = 'DI'
V_IL = 'IL'
V_DC = 'DC'
V_CN = 'CN'

# DATE FORMAT
F_DN_DATE_FMT = '%Y-%m-%d %H:%M:%S'

# AF OUTPUT COLUMNS
F_AF_SUPERLABEL = 'superlabel'
F_AF_BRAND = 'brand'
F_AF_CHANNEL = 'channel'
F_AF_COUNTRY = 'country'
F_AF_DATE = 'date'
F_AF_DATE_M = 'date_m'
F_AF_FORECAST = 'yhat'
F_AF_IS_FORECAST = 'is_forecast'
F_AF_LABEL = 'label'
F_AF_OFFTAKE = 'offtake'
F_AF_OFFTAKE_DC = 'offtake_dc'
F_AF_OFFTAKE_DI = 'offtake_di'
F_AF_OFFTAKE_EIB = 'offtake_eib'
F_AF_OFFTAKE_IL = 'offtake_il'
F_AF_PREDICTION_HORIZON = 'prediction_horizon'
F_AF_RETAILER_INV = 'retailer_inv'
F_AF_RETAILER_INV_COVERAGE = 'retailer_inv_coverage'
F_AF_RETAILER_INV_MONTH = 'retailer_inv_month'
F_AF_SELLIN = 'sellin'
F_AF_SELLIN_DC = 'sellin_dc'
F_AF_SELLIN_DI = 'sellin_di'
F_AF_SELLIN_EIB = 'sellin_eib'
F_AF_SELLIN_IL = 'sellin_il'
F_AF_SELLOUT = 'sellout'
F_AF_SKU = 'sku'
F_AF_SKU_CODE = 'sku_code'
F_AF_SKU_WITH_PKG = 'sku_with_pkg'
F_AF_SKU_WO_PKG = 'sku_wo_pkg'
F_AF_SP_INV = 'sp_inv'
F_AF_SP_INV_DC = 'sp_inv_dc'
F_AF_SP_INV_COVERAGE = 'sp_inv_coverage'
F_AF_SP_INV_MONTH = 'sp_inv_month'
F_AF_STAGE = 'stage'
F_AF_STAGE_3F = 'stage_3f'
F_AF_SUBBRAND = 'tier'

# IMPALA TABLE COLUMNS
F_DN_APO_FLG = 'apo_flg'
F_DN_CRY_COD = 'cry_cod'
F_DN_CYC_DAT = 'cyc_dat'
F_DN_DIS_CHL_COD = 'dis_chl_cod'
F_DN_ETL_TST = 'etl_tst'
F_DN_FRC_CRE_DAT = 'frc_cre_dat'
F_DN_FRC_FLG = 'frc_flg'
F_DN_FRC_MDF_DAT = 'frc_mdf_dat'
F_DN_FRC_USR_NAM_DSC = 'frc_usr_nam_dsc'
F_DN_LV2_UMB_BRD_COD = 'lv2_umb_brd_cod'
F_DN_LV3_PDT_BRD_COD = 'lv3_pdt_brd_cod'
F_DN_LV4_PDT_FAM_COD = 'lv4_pdt_fam_cod'
F_DN_LV5_PDT_SFM_COD = 'lv5_pdt_sfm_cod'
F_DN_LV6_PDT_NAT_COD = 'lv6_pdt_nat_cod'
F_DN_UNIT = 'unit'
F_DN_MAT_COD = 'mat_cod'
F_DN_MEA_DAT = 'mea_dat'
F_DN_OFT_TRK_VAL = 'oft_trk_val'
F_DN_OFT_VAL = 'oft_val'
F_DN_RTL_IVT_COV_VAL = 'rtl_ivt_cov_val'
F_DN_PCK_SKU_COD = 'pck_sku_cod'
F_DN_FRC_MTH_NBR = 'frc_mth_nbr'
F_DN_RTL_IVT_VAL = 'rtl_ivt_val'
F_DN_SAL_INS_VAL = 'sal_ins_val'
F_DN_SAL_OUT_VAL = 'sal_out_val'
F_DN_SUP_IVT_COV_VAL = 'sup_ivt_cov_val'
F_DN_SUP_IVT_VAL = 'sup_ivt_val'
F_DN_USR_NTE_TXT = 'usr_nte_txt'

# AF TO DN MAPPING
MAPPING_AF_TO_DN = {
    F_AF_SUPERLABEL: F_DN_LV2_UMB_BRD_COD,
    F_AF_BRAND: F_DN_LV4_PDT_FAM_COD,
    F_AF_CHANNEL: F_DN_DIS_CHL_COD,
    F_AF_COUNTRY: F_DN_CRY_COD,
    F_AF_DATE: F_DN_MEA_DAT,
    F_AF_DATE_M: F_DN_CYC_DAT,
    F_AF_OFFTAKE: F_DN_OFT_TRK_VAL,
    F_AF_IS_FORECAST: F_DN_FRC_FLG,
    F_AF_LABEL: F_DN_LV3_PDT_BRD_COD,
    F_AF_PREDICTION_HORIZON: F_DN_FRC_MTH_NBR,
    F_AF_RETAILER_INV: F_DN_RTL_IVT_VAL,
    F_AF_RETAILER_INV_COVERAGE: F_DN_RTL_IVT_COV_VAL,
    F_AF_SELLIN: F_DN_SAL_INS_VAL,
    F_AF_SELLOUT: F_DN_SAL_OUT_VAL,
    F_AF_SKU_WO_PKG: F_DN_MAT_COD,
    F_AF_SKU_WITH_PKG: F_DN_PCK_SKU_COD,
    F_AF_SP_INV: F_DN_SUP_IVT_VAL,
    F_AF_SP_INV_COVERAGE: F_DN_SUP_IVT_COV_VAL,
    F_AF_STAGE: F_DN_LV6_PDT_NAT_COD,
    F_AF_SUBBRAND: F_DN_LV5_PDT_SFM_COD,
}

MAPPING_DN_TO_AF = dict()
for key, value in MAPPING_AF_TO_DN.items():
    MAPPING_DN_TO_AF[value] = key

V_DN_LOWEST_GRANULARITY = [
        F_DN_LV2_UMB_BRD_COD, F_DN_LV3_PDT_BRD_COD, F_DN_CRY_COD,
        F_DN_LV4_PDT_FAM_COD, F_DN_LV5_PDT_SFM_COD, F_DN_LV6_PDT_NAT_COD,
        F_DN_MAT_COD, F_DN_PCK_SKU_COD, F_DN_DIS_CHL_COD
    ]
V_AF_LOWEST_GRANULARITY = [MAPPING_DN_TO_AF[key] for key in V_DN_LOWEST_GRANULARITY]

V_ACTUAL = -99
V_CN = 'CN'
V_TINS = 'tins'
V_TONS = 'tons'
