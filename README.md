# <span id="jump1">Danone China International Label Offtake Forecast</span>

Danone-China-International-label-offtake-forecast

Start at Mar.11 2020

---
# <span id="jump2">Project Information</span>
## <span id="jump2.1">Introduction</span>

For IL demand team,they need to do offtake(sales direct by customers) forecast manually every month, which takes them lots of efforts. And the forecast accuracy will dramatically influence the Production Plan.
Such being the case, they wants to have this project to help them forecast offtake using big data and AI.

## <span id="jump2.2">Scope</span>
<!-- TOC -->
- [Danone China International Label Offtake Forecast](#jump1)
- [Project Information](#jump2)
  - [Introduction](#jump2.1)
  - [Scope](#jump2.2)
  - [Folders Description](#jump2.3)
- [Environment Requirements and Setup](#jump3)
- [User Guide](#jump4)
<!-- /TOC -->

## <span id="jump2.3">Folders Description</span>

```
.
|-- README.md           # code description and user guide
|-- requirements.txt    # Python libraries used
|-- cfg                 # Configs, for example data folder path.
|-- run                 # Main function used to trigger the script
|   |-- src             # Functions  used in the controllers 
|-- data                # data storage folder
|-- temp                # folder used to save temporary files
|-- results             # Forecast are generate in this folder

```
---
# <span id="jump3">Environment Requirements and Setup</span>

1. Installed Python3.7 and can execute python script by 'python3 ***.py' in the terminal.
2. Installed pip3 and can install python3.6 package by 'pip3 install ***' in the terminal.
3. Install related python3.7 library, by using following command.
```
pip3 install requirements.txt
```

---

# <span id="jump4">User Guide</span>

In order to generate IL totoal offtake forecast, we can do the following instructions in the terminal

```
cd main
```

```
python3 ForecasterIL.py
```

And the forecast results will be put in the results folder.

