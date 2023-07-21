#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 09:02:47 2023

@author: cghiaus
"""
import pd_dm4bem

TC_file = "TC_tc2ss/TC_tc2ss_sym.csv"
TC = pd_dm4bem.file2TC(TC_file, name="a", auto_number=True)
[Asa, Bsa, Csa, Dsa, ua] = pd_dm4bem.tc2ss(TC)

TC_file = "TC_tc2ss/TC_tc2ss_num.csv"
TC = pd_dm4bem.file2TC(TC_file, name="a")
[Asn, Bsn, Csn, Dsn, un] = pd_dm4bem.tc2ss(TC)

TC_file = "TC_tc2ss/TC_tc2ss_sym.csv"
TC = pd_dm4bem.file2TC(TC_file, name="s")
[Ass, Bss, Css, Dss, us] = pd_dm4bem.tc2ss(TC)
