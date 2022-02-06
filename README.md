# Logistic-RL
# 1. The Problem
The logistic RL environment is created to solve a multi-trip version of
[knapsack problem](https://en.wikipedia.org/wiki/Knapsack_problem) (KP), in which the agent should pack the knapsack 
multiple times and there is also an additional cost for late packing of some of the items. As the first step toward the 
multi-trip knapsack, we implement a single trip version. In the single-trip version you only pack the knapsack once aiming 
at maximum monetary value of the packed items while respecting the maximum capacity of the knapsack.

In particular, we implement a modified version of the bounded knapsack problem (BKP). In BKP, similar to the KP, the
items are coming from a fixed number of categories. Unlike KP, in BKP, the number of available items per each category is
not limited to 0 or 1, but each category can have its unique available number of items. To clarify, consider the 
following example. Imagine we have 3 categories of items to pack, e.g. milk packs, books, oranges:
* In KP, we can either have one or zero of each item, e.g. [0, 1, 1] meaning no milk pack, 1 book and 1 orange to choose
from.
* In BKP, we can have a variable number of items between zero and a category dependent maximum, e.g. [0, 10, 2] 
corresponding to zero milk pack, 10 books, and 2 oranges, where maximum for different categories in this case 
could be [5, 20, 10] which indicates that we can have maximum 5 milk packs, 20 books, and 10 oranges.

## 1.1. The Objective
As mentioned above, the objective to maximize the total monetary value of packed items given that the total mass of the 
packed items should be below a predefined threshold (i.e. the knapsack's capacity). 

## 1.2. State, Actions, Rewards, Termination

### 1.2.1. State
In this environment, at each step, the agent decides based on the following observations (i.e. **state**):
* the used capacity up to this step,
* the number of available items in each category.
### 1.2.2. Action
At each step, the **action** is choosing one of the categories to pick an item from for packing. 

If the agent takes an action which is not valid, i.e. choosing a category in which no item is left (for example,
choosing to pack milk in the above example):
* the action is not considered, 
* the state is not changed,
* the reward of zero is returned, and 
* the process continues.

### 1.2.3. Reward
The **reward** that the agent receives is the monetary value of the packed item, if that action is accepted and does 
not lead to termination. In the latter cases, the reward is set to zero. 

### 1.2.4. Termination
An episode is terminated when 
* the agent attempts going beyond the available mass capacity. In that case:
    - the state is not affected with this last action,
    - the episode is ended,
    - reward zero is returned
* when all the available items are gone (there is nothing left to pick up).

## 2. Installation
### 2.1. Creating the conda env
```buildoutcfg
conda create -n knapsack python=3.8
```

### 2.2. Installing the required packages
Activate the conda env:
```buildoutcfg
conda activate knapsack
```
### 2.3. Update your pip
```buildoutcfg
pip install --upgrade pip
```
### 2.4. Clone the project
```buildoutcfg
git clone git@github.com:nima-siboni/Logistic-RL.git
cd Logistic-RL
```

### 2.5. Install the requirements
Install the requirements in the env
```buildoutcfg
pip install -r requirements.txt
```

### 2.6. Install the package locally
Install the Logistic-RL package  in the current environment
```buildoutcfg
pip install -e .
```

Have fun!

## 3. How to use the package
### 3.1. Environment config
In order to create an environment, one needs a config file, e.g. [meta_conf.conf](./meta_conf.conf):
```python
{
    Env {
        capacity = 70 # Total mass capacity of the knapsack
        nr_categories = 10 # Number of categories
        specific_masses = [0.01, 0.02, 0.05, 0.50, 1.00, 2.00, 5.00, 10.0, 20.0, 31.50] # in Kg similar to DHL (2019)
        specific_values  = [0.45, 0.80, 0.95, 1.55, 2.70, 3.79, 4.50, 6.99, 9.49, 16.49] # in Euros, like above.
        seed_value = None  # if None is given it randomly chooses a seed.
        min_availability = [10,   10,  10,  10, 5,  2,  2,  2, 1, 1] # min number of possible items for each category
        max_availability = [100, 100, 100, 100, 50, 25, 20, 5, 3, 2] # max number of possible items for each category
    }
}
```
Let's have a more detailed look at the necessary ingredients of an env config:
* ```capcaity```: the total mass capacity of the knapsack. This is our constraint on mass.
* ```nr_categories```: the number of categories to choose items from. In the example above with milk packs, books, and 
oranges, the number of categories was equal to 3.
* ```specific_masses```: a list composed of the weights corresponding to each category; e.g. here the items in the first
category have mass of 10 gr.
* ```specific_values```: a list made of monetary values the agent earn for choosing each category; e.g. by choosing an 
item from the first category, the agent earn 45 cents.
* ```seed```: the random seed for the setting the initial list of number of items in each category.
* ```min_ and max_availability```: when initializing the environment, a random list of available items is created, where the number
of available items for each category is an integer between the min and max availability. For example for the above 
config, the initial number of items available in the first four categories are in range [10, 100).

Note: The above values for mass and prices are based on
(DHL Leistungen und Preise)[https://www.deutschepost.de/content/dam/dpag/images/G_g/Gesamtpreisliste/dp-leistungen-und-preise-01-2022.pdf]
### 3.2 -- Instantiating the Environment
After creating the config file, one can instantiate an instance of the environment simply by:
```python
from pyhocon import ConfigFactory
from environment.knapsack import Knapsack
from utils.utils import conf_value_checking

# Reading the configs
conf = ConfigFactory.parse_file('meta_conf.conf')
# Checking the sanity of configs
conf_value_checking(conf)

# Creating the environment
env = Knapsack(env_config={'conf': conf.Env})
```
which is also done in [training.py](./training.py) .
