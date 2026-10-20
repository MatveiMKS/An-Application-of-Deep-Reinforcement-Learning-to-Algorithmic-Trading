# coding=utf-8

"""
Goal: Implement a trading environment compatible with OpenAI Gym.
Authors: Thibaut Théate and Damien Ernst
Institution: University of Liège
"""

###############################################################################
################################### Imports ###################################
###############################################################################

import os
import gym
import math
import numpy as np

import pandas as pd
pd.options.mode.chained_assignment = None

from matplotlib import pyplot as plt

from dataDownloader import AlphaVantage
from dataDownloader import YahooFinance
from dataDownloader import CSVHandler
from fictiveStockGenerator import StockGenerator



###############################################################################
################################ Global variables #############################
###############################################################################

# Boolean handling the saving of the stock market data downloaded
saving = True

# Variable related to the fictive stocks supported
fictiveStocks = ('LINEARUP', 'LINEARDOWN', 'SINUSOIDAL', 'TRIANGLE')

positionMode = 'longShort'
positionModes = ('longShort', 'longShortFunding', 'longCash')
dailyShortCost = 0.0003
shortEpsilon = 'auto'
fractionalShares = True


###############################################################################
############################## Class TradingEnv ###############################
###############################################################################

class TradingEnv(gym.Env):
    """
    GOAL: Implement a custom trading environment compatible with OpenAI Gym.
    
    VARIABLES:  - data: Dataframe monitoring the trading activity.
                - state: RL state to be returned to the RL agent.
                - reward: RL reward to be returned to the RL agent.
                - done: RL episode termination signal.
                - t: Current trading time step.
                - marketSymbol: Stock market symbol.
                - startingDate: Beginning of the trading horizon.
                - endingDate: Ending of the trading horizon.
                - stateLength: Number of trading time steps included in the state.
                - numberOfShares: Number of shares currently owned by the agent.
                - transactionCosts: Transaction costs associated with the trading
                                    activity (e.g. 0.01 is 1% of loss).
                                
    METHODS:    - __init__: Object constructor initializing the trading environment.
                - reset: Perform a soft reset of the trading environment.
                - step: Transition to the next trading time step.
                - render: Illustrate graphically the trading environment.
    """

    def __init__(self, marketSymbol, startingDate, endingDate, money, stateLength=30,
                 transactionCosts=0, startingPoint=0):
        """
        GOAL: Object constructor initializing the trading environment by setting up
              the trading activity dataframe as well as other important variables.
        
        INPUTS: - marketSymbol: Stock market symbol.
                - startingDate: Beginning of the trading horizon.
                - endingDate: Ending of the trading horizon.
                - money: Initial amount of money at the disposal of the agent.
                - stateLength: Number of trading time steps included in the RL state.
                - transactionCosts: Transaction costs associated with the trading
                                    activity (e.g. 0.01 is 1% of loss).
                - startingPoint: Optional starting point (iteration) of the trading activity.
        
        OUTPUTS: /
        """

        # CASE 1: Fictive stock generation
        if(marketSymbol in fictiveStocks):
            stockGeneration = StockGenerator()
            if(marketSymbol == 'LINEARUP'):
                self.data = stockGeneration.linearUp(startingDate, endingDate)
            elif(marketSymbol == 'LINEARDOWN'):
                self.data = stockGeneration.linearDown(startingDate, endingDate)
            elif(marketSymbol == 'SINUSOIDAL'):
                self.data = stockGeneration.sinusoidal(startingDate, endingDate)
            else:
                self.data = stockGeneration.triangle(startingDate, endingDate)
 
        # CASE 2: Real stock loading
        else:
            # Check if the stock market data is already present in the database
            csvConverter = CSVHandler()
            csvName = "".join(['Data/', marketSymbol, '_', startingDate, '_', endingDate])
            exists = os.path.isfile(csvName + '.csv')
            
            # If affirmative, load the stock market data from the database
            if(exists):
                self.data = csvConverter.CSVToDataframe(csvName)
            # Otherwise, download the stock market data from Yahoo Finance and save it in the database
            else:  
                downloader1 = YahooFinance()
                downloader2 = AlphaVantage()
                try:
                    self.data = downloader1.getDailyData(marketSymbol, startingDate, endingDate)
                except:
                    self.data = downloader2.getDailyData(marketSymbol, startingDate, endingDate)

                if saving == True:
                    csvConverter.dataframeToCSV(csvName, self.data)

        # Interpolate in case of missing data
        self.data.replace(0.0, np.nan, inplace=True)
        self.data.interpolate(method='linear', limit=5, limit_area='inside', inplace=True)
        self.data.fillna(method='ffill', inplace=True)
        self.data.fillna(method='bfill', inplace=True)
        self.data.fillna(0, inplace=True)
        
        # Set the trading activity dataframe
        self.data['Position'] = 0
        self.data['Action'] = 0
        self.data['Holdings'] = 0.
        self.data['Cash'] = float(money)
        self.data['Money'] = self.data['Holdings'] + self.data['Cash']
        self.data['Returns'] = 0.

        # Set the RL variables common to every OpenAI gym environments
        self.state = [self.data['Close'][0:stateLength].tolist(),
                      self.data['Low'][0:stateLength].tolist(),
                      self.data['High'][0:stateLength].tolist(),
                      self.data['Volume'][0:stateLength].tolist(),
                      [0]]
        self.reward = 0.
        self.done = 0

        # Set additional variables related to the trading activity
        self.marketSymbol = marketSymbol
        self.startingDate = startingDate
        self.endingDate = endingDate
        self.stateLength = stateLength
        self.t = stateLength
        self.numberOfShares = 0
        self.transactionCosts = transactionCosts

        if positionMode not in positionModes:
            raise SystemExit("Unknown position mode: " + str(positionMode))
        self.positionMode = positionMode
        if shortEpsilon == 'auto':
            self.epsilon = self.maximumPriceIncrease()
        else:
            self.epsilon = shortEpsilon

        # If required, set a custom starting point for the trading activity
        if startingPoint:
            self.setStartingPoint(startingPoint)


    def reset(self):
        """
        GOAL: Perform a soft reset of the trading environment. 
        
        INPUTS: /    
        
        OUTPUTS: - state: RL state returned to the trading strategy.
        """

        # Reset the trading activity dataframe
        self.data['Position'] = 0
        self.data['Action'] = 0
        self.data['Holdings'] = 0.
        self.data['Cash'] = self.data['Cash'][0]
        self.data['Money'] = self.data['Holdings'] + self.data['Cash']
        self.data['Returns'] = 0.

        # Reset the RL variables common to every OpenAI gym environments
        self.state = [self.data['Close'][0:self.stateLength].tolist(),
                      self.data['Low'][0:self.stateLength].tolist(),
                      self.data['High'][0:self.stateLength].tolist(),
                      self.data['Volume'][0:self.stateLength].tolist(),
                      [0]]
        self.reward = 0.
        self.done = 0

        # Reset additional variables related to the trading activity
        self.t = self.stateLength
        self.numberOfShares = 0

        return self.state

    
    def computeLowerBound(self, cash, numberOfShares, price):
        """
        GOAL: Compute the lower bound of the complete RL action space, 
              i.e. the minimum number of share to trade.
        
        INPUTS: - cash: Value of the cash owned by the agent.
                - numberOfShares: Number of shares owned by the agent.
                - price: Last price observed.
        
        OUTPUTS: - lowerBound: Lower bound of the RL action space.
        """

        # Computation of the RL action lower bound
        deltaValues = - cash - numberOfShares * price * (1 + self.epsilon) * (1 + self.transactionCosts)
        if deltaValues < 0:
            lowerBound = deltaValues / (price * (2 * self.transactionCosts + (self.epsilon * (1 + self.transactionCosts))))
        else:
            lowerBound = deltaValues / (price * self.epsilon * (1 + self.transactionCosts))
        return lowerBound
    

    def maximumPriceIncrease(self):
        """
        GOAL: Compute the largest relative increase of the close price between
              two consecutive time steps of the trading data (ADAPTATION BTC).
        
        INPUTS: /
        
        OUTPUTS: - epsilon: Largest relative price increase (at least 0.1, as
                            in the original code).
        """

        increase = self.data['Close'].pct_change().max()
        if np.isnan(increase):
            increase = 0
        return max(0.1, float(increase))


    def quantity(self, x):
        """
        GOAL: Round a number of shares, down to an integer if fractional
              quantities are not allowed (ADAPTATION BTC).
        """

        return x if fractionalShares else math.floor(x)


    def computeTransition(self, action, t, numberOfShares):
        """
        GOAL: Compute the outcome of a trading decision at time step t, without
              modifying the environment. Used for both the action executed and
              the other action (exploration trick), which fixes an error of the
              original code where the other action used the updated number of
              shares instead of the previous one (ADAPTATION).
        
        INPUTS: - action: Trading decision (1 = long, 0 = short or cash).
                - t: Current trading time step.
                - numberOfShares: Number of shares owned before the decision.
        
        OUTPUTS: - position: New trading position (1, -1 or 0).
                 - cash: New cash value.
                 - holdings: New value of the shares owned.
                 - numberOfShares: New number of shares owned.
                 - customReward: Whether the reward is the special one of a
                                 forced partial closing of a short position.
        """

        price = self.data['Close'][t]
        previousPrice = self.data['Close'][t-1]
        previousPosition = self.data['Position'][t-1]
        cash = self.data['Cash'][t-1]
        customReward = False

        # ADAPTATION BTC : cost of borrowing the shares of a short position
        if self.positionMode == 'longShortFunding' and previousPosition == -1:
            cash -= dailyShortCost * numberOfShares * previousPrice

        # CASE 1: LONG POSITION
        if(action == 1):
            position = 1
            # Case a: Long -> Long
            if(previousPosition == 1):
                pass
            # Case b: No position -> Long
            elif(previousPosition == 0):
                numberOfShares = self.quantity(cash/(price * (1 + self.transactionCosts)))
                cash = cash - numberOfShares * price * (1 + self.transactionCosts)
            # Case c: Short -> Long
            else:
                cash = cash - numberOfShares * price * (1 + self.transactionCosts)
                numberOfShares = self.quantity(cash/(price * (1 + self.transactionCosts)))
                cash = cash - numberOfShares * price * (1 + self.transactionCosts)
            holdings = numberOfShares * price

        # CASE 2: CASH POSITION (ADAPTATION BTC, 'longCash' mode)
        elif(action == 0 and self.positionMode == 'longCash'):
            position = 0
            # Case a: Long -> Cash
            if(previousPosition == 1):
                cash = cash + numberOfShares * price * (1 - self.transactionCosts)
            numberOfShares = 0
            holdings = 0.

        # CASE 3: SHORT POSITION
        elif(action == 0):
            position = -1
            # Case a: Short -> Short
            if(previousPosition == -1):
                lowerBound = self.computeLowerBound(cash, -numberOfShares, previousPrice)
                if lowerBound > 0:
                    numberOfSharesToBuy = min(self.quantity(lowerBound), numberOfShares)
                    numberOfShares -= numberOfSharesToBuy
                    cash = cash - numberOfSharesToBuy * price * (1 + self.transactionCosts)
                    customReward = True
            # Case b: No position -> Short
            elif(previousPosition == 0):
                numberOfShares = self.quantity(cash/(price * (1 + self.transactionCosts)))
                cash = cash + numberOfShares * price * (1 - self.transactionCosts)
            # Case c: Long -> Short
            else:
                cash = cash + numberOfShares * price * (1 - self.transactionCosts)
                numberOfShares = self.quantity(cash/(price * (1 + self.transactionCosts)))
                cash = cash + numberOfShares * price * (1 - self.transactionCosts)
            holdings = - numberOfShares * price

        # CASE 4: PROHIBITED ACTION
        else:
            raise SystemExit("Prohibited action! Action should be either 1 (long) or 0 (short or cash).")

        return position, cash, holdings, numberOfShares, customReward


    def computeReward(self, t, money, customReward):
        """
        GOAL: Compute the RL reward, as in the original code: the relative
              change of the portfolio value, except for a forced partial
              closing of a short position.
        """

        if not customReward:
            return (money - self.data['Money'][t-1])/self.data['Money'][t-1]
        return (self.data['Close'][t-1] - self.data['Close'][t])/self.data['Close'][t-1]


    def step(self, action):
        """
        GOAL: Transition to the next trading time step based on the
              trading position decision made (either long or short/cash).
        
        INPUTS: - action: Trading decision (1 = long, 0 = short or cash).    
        
        OUTPUTS: - state: RL state to be returned to the RL agent.
                 - reward: RL reward to be returned to the RL agent.
                 - done: RL episode termination signal (boolean).
                 - info: Additional information returned to the RL agent.
        """

        # Stting of some local variables
        t = self.t
        numberOfShares = self.numberOfShares

        # Transition related to the action executed
        position, cash, holdings, self.numberOfShares, customReward = self.computeTransition(action, t, numberOfShares)
        if position != self.data['Position'][t-1]:
            self.data['Action'][t] = 1 if position == 1 else -1
        self.data['Position'][t] = position
        self.data['Cash'][t] = cash
        self.data['Holdings'][t] = holdings

        # Update the total amount of money owned by the agent, as well as the return generated
        self.data['Money'][t] = self.data['Holdings'][t] + self.data['Cash'][t]
        self.data['Returns'][t] = (self.data['Money'][t] - self.data['Money'][t-1])/self.data['Money'][t-1]

        # Set the RL reward returned to the trading agent
        self.reward = self.computeReward(t, self.data['Money'][t], customReward)

        # Transition to the next trading time step
        self.t = self.t + 1
        self.state = [self.data['Close'][self.t - self.stateLength : self.t].tolist(),
                      self.data['Low'][self.t - self.stateLength : self.t].tolist(),
                      self.data['High'][self.t - self.stateLength : self.t].tolist(),
                      self.data['Volume'][self.t - self.stateLength : self.t].tolist(),
                      [self.data['Position'][self.t - 1]]]
        if(self.t == self.data.shape[0]):
            self.done = 1  

        # Same reasoning with the other action (exploration trick)
        otherAction = int(not bool(action))
        otherPosition, otherCash, otherHoldings, _, otherCustomReward = self.computeTransition(otherAction, t, numberOfShares)
        otherReward = self.computeReward(t, otherHoldings + otherCash, otherCustomReward)
        otherState = [self.data['Close'][self.t - self.stateLength : self.t].tolist(),
                      self.data['Low'][self.t - self.stateLength : self.t].tolist(),
                      self.data['High'][self.t - self.stateLength : self.t].tolist(),
                      self.data['Volume'][self.t - self.stateLength : self.t].tolist(),
                      [otherPosition]]
        self.info = {'State' : otherState, 'Reward' : otherReward, 'Done' : self.done}

        # Return the trading environment feedback to the RL trading agent
        return self.state, self.reward, self.done, self.info

    def render(self):
        """
        GOAL: Illustrate graphically the trading activity, by plotting
              both the evolution of the stock market price and the 
              evolution of the trading capital. All the trading decisions
              (long and short positions) are displayed as well.
        
        INPUTS: /   
        
        OUTPUTS: /
        """

        # Set the Matplotlib figure and subplots
        fig = plt.figure(figsize=(10, 8))
        ax1 = fig.add_subplot(211, ylabel='Price', xlabel='Time')
        ax2 = fig.add_subplot(212, ylabel='Capital', xlabel='Time', sharex=ax1)

        # Plot the first graph -> Evolution of the stock market price
        self.data['Close'].plot(ax=ax1, color='blue', lw=2)
        ax1.plot(self.data.loc[self.data['Action'] == 1.0].index, 
                 self.data['Close'][self.data['Action'] == 1.0],
                 '^', markersize=5, color='green')   
        ax1.plot(self.data.loc[self.data['Action'] == -1.0].index, 
                 self.data['Close'][self.data['Action'] == -1.0],
                 'v', markersize=5, color='red')
        
        # Plot the second graph -> Evolution of the trading capital
        self.data['Money'].plot(ax=ax2, color='blue', lw=2)
        ax2.plot(self.data.loc[self.data['Action'] == 1.0].index, 
                 self.data['Money'][self.data['Action'] == 1.0],
                 '^', markersize=5, color='green')   
        ax2.plot(self.data.loc[self.data['Action'] == -1.0].index, 
                 self.data['Money'][self.data['Action'] == -1.0],
                 'v', markersize=5, color='red')
        
        # Generation of the two legends and plotting
        ax1.legend(["Price", "Long",  "Short"])
        ax2.legend(["Capital", "Long", "Short"])
        plt.savefig(''.join(['Figures/', str(self.marketSymbol), '_Rendering', '.png']))
        #plt.show()


    def setStartingPoint(self, startingPoint):
        """
        GOAL: Setting an arbitrary starting point regarding the trading activity.
              This technique is used for better generalization of the RL agent.
        
        INPUTS: - startingPoint: Optional starting point (iteration) of the trading activity.
        
        OUTPUTS: /
        """

        # Setting a custom starting point
        self.t = np.clip(startingPoint, self.stateLength, len(self.data.index))

        # Set the RL variables common to every OpenAI gym environments
        self.state = [self.data['Close'][self.t - self.stateLength : self.t].tolist(),
                      self.data['Low'][self.t - self.stateLength : self.t].tolist(),
                      self.data['High'][self.t - self.stateLength : self.t].tolist(),
                      self.data['Volume'][self.t - self.stateLength : self.t].tolist(),
                      [self.data['Position'][self.t - 1]]]
        if(self.t == self.data.shape[0]):
            self.done = 1
    