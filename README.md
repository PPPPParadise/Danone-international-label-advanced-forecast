# Danone China Supply Chain Automation

Danone-China-International-label-offtake-forecast

Start at Mar.11 2020

---
# Project Information
## Introduction

For IL demand team,they need to do offtake(sales direct by customers) forecast manually every month, 
which takes them lots of efforts. And the forecast accuracy will dramatically influence the Production Plan.
Such being the case, they wants to have this project to help them forecast offtake using big data and AI.

## Scope
<!-- TOC -->

- [Danone China Supply Chain Automation](#danone-china-supply-chain-automation)
- [Project Information](#project-information)
  - [Introduction](#introduction)
  - [Scope](#scope)
  - [Folders Description](#folders-description)
- [Environment Requirements and Setup](#environment-requirements-and-setup)
- [User Guide](#user-guide)
  - [Handover developer requirement](#handover-developer-requirement)
  - [Flowchart](#flowchart)
  - [Implement](#implement)
- [Naming Rules](#naming-rules)
  - [Naming rules in the controllers folder](#naming-rules-in-the-controllers-folder)
  - [Naming rules in the data/structured_data folder](#naming-rules-in-the-structured-data-folder)
       
<!-- /TOC -->

## Folders Description

```
.
|-- README.md           # code description and user guide
|-- requirements.txt    # Python libraries used
|-- config              # Configs, for example data folder path.
|-- controllers         # Main function used to trigger the script
|   |-- src             # Functions  used in the controllers 
|-- data                # data storage folder
|   |-- input_data      # input data
|   |-- logs            # script's running logs
|-- temp                # folder used to save temporary files
`-- results             # Forecast are generate in this folder

```
---
# Environment Requirements and Setup
1. Installed Python3.6 and can execute python script by 'python3 ***.py' in the terminal.
2. Installed pip3 and can install python3.6 package by 'pip3 install ***' in the terminal.
3. Install related python3.6 library, by using following command.
```
pip3 install requirements.txt
```

---
# User Guide

```

