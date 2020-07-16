#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
pd.set_option('float_format', lambda x: '%.2f' % x)
import warnings
warnings.filterwarnings('ignore') 


# In[2]:


class KPI_formalization:
    
    def __init__(self, KPI_input,kpi_name = 'bias',skip_month = '2019-06',version = 'AF_BCG'):
        
        self.accuracy = KPI_input.\
            query("kpi_name == '%s'" %'fa').\
            query("horizon == '1'").\
            query("level == 'all_skus'").\
            query("kpi_type == 'ytd'").\
            query("kpi_type == 'ytd'").\
            query("month == '2019-12'").\
            reset_index(drop = True)
        
        self.KPI = KPI_input.\
                   query("kpi_name == '%s'" %'bias').\
                   query("(month != '2019-06')&(month != '2019-04')&(month != '2019-05')").\
                   reset_index(drop = True)
        self.version = version
        self.country_brand_ratio = pd.DataFrame({'country_brand':['NL_NC', 'ANZ_KC', 'UK_C&G', 'UK_APT', 'DE_APT', 'ANZ_APT'],
                                                 'Share': [0.11, 0.05, 0.04, 0.07 ,0.18 ,0.54]})
        self.horizon_ratio = pd.DataFrame({'horizon':['1', '4', 'R6M'],
                                           'weight': [0.2, 0.6, 0.2]})
        
        
    def merge_volume_ratio(self,brand_month):
        
        # Merge volume ratio
        brand_month = pd.merge(brand_month.reset_index(),
                               self.country_brand_ratio,
                               left_on = [('computed_on','')],
                               right_on = 'country_brand',
                               how = 'left')
        return brand_month
    
    
    def merge_index_weight(self,brand_month):
        
        # Merge volume ratio
        brand_month = pd.merge(brand_month,
                               self.horizon_ratio,
                               left_on = [('horizon','')],
                               right_on = 'horizon',
                               how = 'left')
        return brand_month

    
    def calculate_brand_month(self):
        
        brand_month = pd.pivot_table(self.KPI.query("level == '%s'" %'country_brand').query("kpi_type == '%s'" %'month').query("version == '%s'" %self.version),
                                     index = ['horizon','computed_on'],
                                     columns = ['month'],
                                     values = ['value'])
        # Calculate Variance
        brand_month['variance'] = brand_month.apply(lambda x: np.sqrt((x**2).sum()/brand_month.shape[1]), axis=1)
        
        # Merge Country_brand share and Weighted Share
        brand_month = self.merge_volume_ratio(brand_month)
        brand_month = self.merge_index_weight(brand_month)
        
        # Calculate brand_month index
        brand_month['brand_month_index'] = brand_month[('variance','')] * brand_month['Share'] * brand_month['weight']* 10
        # Return index 
        print ('KPI :',' brand_month = ','%.1f' % brand_month['brand_month_index'].sum())
        brand_month = brand_month['brand_month_index'].sum()
        return brand_month
    
    
    def calculate_brand_ytd(self):
        
        brand_ytd = pd.pivot_table(self.KPI.query("level == '%s'" %'country_brand').query("kpi_type == '%s'" %'ytd').query("version == '%s'" %self.version).query("month == '%s'" %'2019-12'),
                                   index = ['horizon','computed_on'],
                                   columns = ['month'],
                                   values = ['value'])
        
        # Merge Country_brand share and Weighted Share
        brand_ytd = self.merge_volume_ratio(brand_ytd)
        brand_ytd = self.merge_index_weight(brand_ytd)
        
        # Calculate brand_month index
        brand_ytd['brand_ytd_index'] = np.abs(brand_ytd[(('value', '2019-12'))] * brand_ytd['Share'] * brand_ytd['weight']* 10)
        print ('KPI :',' brand_ytd = ','%.1f' % brand_ytd['brand_ytd_index'].sum())
        brand_ytd = brand_ytd['brand_ytd_index'].sum()
        return brand_ytd
        
    def calculate_ttl_ytd(self):
        
        ttl_ytd = pd.pivot_table(self.KPI.query("level == '%s'" %'all_skus').query("kpi_type == '%s'" %'ytd').query("version == '%s'" %self.version).query("month == '%s'" %'2019-12'),
                                 index = ['horizon','computed_on'],
                                 columns = ['month'],
                                 values = ['value']).reset_index()
        
        # Merge Weighted Share
        ttl_ytd = self.merge_index_weight(ttl_ytd)
        
        # Calculate brand_month index
        ttl_ytd['ttl_ytd_index'] = np.abs(ttl_ytd[(('value', '2019-12'))] * ttl_ytd['weight'])
        print ('KPI :',' ttl_ytd = ','%.1f' % ttl_ytd['ttl_ytd_index'].sum())
        ttl_ytd = ttl_ytd['ttl_ytd_index'].sum()
        return ttl_ytd
        
        
    def calculate_ttl_month(self):
        
        ttl_month = pd.pivot_table(self.KPI.query("level == '%s'" %'all_skus').query("kpi_type == '%s'" %'month').query("version == '%s'" %self.version),
                                   index = ['horizon','computed_on'],
                                   columns = ['month'],
                                   values = ['value'])

        # Calculate Variance
        ttl_month['variance'] = ttl_month.apply(lambda x: np.sqrt((x**2).sum()/ttl_month.shape[1]), axis=1)
        ttl_month = ttl_month.reset_index()
        
        # Merge Weighted Share
        ttl_month = self.merge_index_weight(ttl_month)
        
        # Calculate ttl_month index
        ttl_month['ttl_month_index'] = np.abs(ttl_month[('variance','')] * ttl_month['weight'])
        print ('KPI :',' ttl_month = ','%.1f' % ttl_month['ttl_month_index'].sum())
        ttl_month = ttl_month['ttl_month_index'].sum()
        return ttl_month
    
    def calculate_ttl_FA(self):
        return self.accuracy['value'].sum()
    
    def generate_KPI(self):
        
        print ('Model :','version = ',self.version)
        brand_month = self.calculate_brand_month()
        brand_ytd = self.calculate_brand_ytd()
        ttl_month = self.calculate_ttl_month()
        ttl_ytd = self.calculate_ttl_ytd()
        ttl_score = 0.3 * 0.7 * brand_month + 0.7 * 0.7 * brand_ytd + 0.3 * 0.7 * ttl_ytd + 0.3 * 0.3 * ttl_month
        print ('KPI_score : ',ttl_score)
        print ('accuracy : ',ttl_fa)
        return ttl_score
        


# #### Read input Files

# In[3]:


#df = pd.read_csv('kpi_brand_model_comparison.csv')


# #### BCG_model

# In[4]:


#kpi = KPI_formalization(df,version = 'AF_BCG')


# In[5]:


#kpi.generate_KPI()

