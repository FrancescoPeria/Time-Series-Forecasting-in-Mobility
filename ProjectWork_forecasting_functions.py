
import numpy as np
from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt


#####################################################################################################################################

# Calcolo statistiche delle varie serie temporali

def calc_stats(A, start, end):

    mask = (A.index >= start) & (A.index < end)
    A = A[mask]
    Horizon = pd.Series(np.full(A.shape[1], A.shape[0]), index = A.columns)
    NZ_buckets = A[A!=0].count(axis = 0)
    
    # Media e Deviazione_Std --> considerati solo i periodi con osservazioni != 0
    mu = A[A!=0].apply(lambda x: x.mean(axis = 0))
    std = A[A!=0].apply(lambda x: x.std(axis = 0))
    
    # ADI e CV
    ADI = Horizon / NZ_buckets
    CV = std / mu 
    CV_squared = CV**2
    
    # Dò nome alle serie
    Horizon.name = 'Horizon'
    NZ_buckets.name = 'NZ_buckets'
    mu.name = 'Mean'
    std.name = 'Std'
    ADI.name = 'ADI'
    CV.name = 'CV'
    CV_squared.name = 'CV2'
    
    return(Horizon, NZ_buckets, mu, std, ADI, CV, CV_squared)


# Segmentazione sulla base delle statistiche di cui sopra

def assign_segment(A):

    x = None
    
    # Classica categorizzazione su ADI e CV
    if (A['CV2'] <= 0.49) and (A['ADI'] <= 1.32):
        x = 'SMOOTH'
    elif (A['CV2'] <= 0.49) and (A['ADI'] > 1.32):
        x = 'INTERMITTENT'
    elif (A['CV2'] > 0.49) and (A['ADI'] <= 1.32):
        x = 'ERRATIC'
    else:
        x = 'LUMPY'
    
    return(x)




#####################################################################################################################################

# Definizione della funzione con PROMO per creare (X_train, y_train, X_test, y_test)
def create_dataframe(A, xlen, ylen, test_loops):
    
    periods_considered = A.columns
    print('='*60)
    print('1st fitted value on the training set : ', periods_considered[xlen])
    print('Last fitted value on the training set : ', periods_considered[-test_loops-1])
    print('='*60)
    
    D = A.values # D che prima memorizzava un dataframe, adesso memorizza un 2D-array
    rows, periods = D.shape
    
    # Creo il dataset G dei giorni riferiti alle date relative alle colonne di D
    # Per far funzionare questa funzione ogni colonna di D deve essere un datetime
    
    # 0=Sunday, 6=Saturday
    giorno = [ int( col.strftime('%w') ) + 1 for col in A.columns]
    giorno = np.reshape(giorno, [1, -1])
    G = np.repeat( giorno, rows, axis = 0)
    
    """" G è fatto in questo modo
    [[G1, G2, ..., Gperiod_considered],
     [G1, G2, ..., Gperiod_considered],
     [G1, G2, ..., Gperiod_considered]
     ...] ogni riga ripetuta rows volte
    """ 
    
    # Stessa cosa per l'ora all'interno della giornata
    ora = [ int( col.strftime('%H') ) for col in A.columns]
    ora = np.reshape(ora, [1, -1])
    O = np.repeat( ora, rows, axis = 0)
    
    #===============================================================================================================================
    
    # 1) CREAZIONE TRAIN
    
    loops = periods - xlen - ylen + 1
    
    train = [] #diventerà una lista di array che vengono appesi con il ciclo sottostante
    
    # A ogni iterazione appendo un array composto da tutte le righe di A ma solo xlen+ylen
    # colonne di volta in volta spostandosi a destra di 1 colonna
    for col in range(loops): 
        
        # Finestra del dataset che contiene xlen-features e che contiene ylen punti da predire
        d = D[ : , col : col+xlen+ylen ]
        
        # Ci metto il primo dei giorni e la prima delle ore che devo andare a prevedere,
        # per avere un riferimento temporale nell'arco della settimana
        g = G[ : , col+xlen].reshape(-1, 1)
        o = O[ : , col+xlen].reshape(-1, 1)
        
        # Stacking orizzontale per creare il dataset da dare in input a un qualsiasi modello di ML
        train.append(np.hstack([g, o, d]))
            
    # Da una lista di array creo un unico 2d-array dove ogni riga è una forecasting unit e 
    # ogni colonna è un periodo, ovvero un lag, variabile a seconda della semantica della feature
    train = np.vstack(train)
    
    # Splitto il mio 2d-array in 2 array: X_train e y_train. Nell'argomento di split viene 
    # specificato il numero di colonne che deve avere y_train, ovvero ylen e per differenza 
    # X_train sarà una matrice con le colonne residuali
    X_train, y_train = np.split(train, [-ylen], axis = 1)
    
    #===============================================================================================================================
    
    # 2) CREAZIONE TEST se test_loops > 0 --> riprevedo il passato facendo backtesting
    
    # Se voglio fare un BACKTEST (usare una frazione degli historical data par come TEST) 
    # e quindi rinunciare a una parte delle osservazioni in fase di training per poi usarle
    # in fase di test, splitto X_train in un X_train più piccolo e in X_test.
    # La stessa identica cosa faccio con y_train
    if test_loops > 0: 
        
        X_train, X_test = np.split(X_train, [-rows*test_loops], axis = 0)
        y_train, y_test = np.split(y_train, [-rows*test_loops], axis = 0)
        print('='*60)
        print('1st predictable test period : ', periods_considered[-test_loops-ylen+1])
        print('Last predictable test period : ', periods_considered[-1])
        print('='*60)                  
     
    #===============================================================================================================================   
    
    # 3) Se invece non devo fare il TEST e quindi se NON devo valutare il modello sul TEST set (backtesting)
    #    ma mi serve solo fare FCST futura
    
    # Se invece non voglio fare validazione ma solo forecast futura, utilizzo le ultime xlen 
    # colonne di D per fare una vera e propria forecast nel futuro, che non può essere valutata
    # in quanto non esiste lo storico a fronte del quale calcolare l'accuracy
    else: 
        
        # ---------------------------------------------------------------------
        
        # Inserisco nel test il primo giorno che vado a predire prendendolo da argomento della funzione
        data_fcst_futura = periods_considered[-1] + timedelta(hours = 1)
        giorno_fcst_futura = int( data_fcst_futura.strftime('%w') )
        ora_fcst_futura = int( data_fcst_futura.strftime('%H') )
        
        X_giorno = np.full((rows, 1), giorno_fcst_futura)
        X_ora = np.full((rows, 1), ora_fcst_futura)
            
        # ---------------------------------------------------------------------
            
        # Adesso mi ricreo il dataset di test da usare per model.predict(X_test)
        
        X_test = np.hstack( [X_giorno, X_ora, D[:, -xlen:]] ) # uso le ultime xlen colonne per fare una previsione
        
        # y_test fatto di NaN ha le stesse righe di X_test e ylen colonne
        y_test = np.full((rows, ylen), np.nan)
        
    #===============================================================================================================================     
    
    feature_names = ( ['giorno_FCST', 'ora_FCST'] + 
                      ['lag_' + str(j) for j in range(xlen, 0, -1)] )  
    

    return (X_train, y_train, X_test, y_test, np.array(feature_names) )



####################################################################################################################################

# Dataset con metriche di errore a confronto con TRAIN e TEST

def ML_forecast_KPI(y_train, y_train_pred, y_test, y_pred):
    
    A = pd.DataFrame(columns = ['MAE%', 'RMSE%', 'BIAS%', 'R^2'], index = ['Train', 'Test'])
    
    # COME IL MODELLO FITTA I DATI
    # Ricorda che in time series forecasting, la performance previsionale sarà buona 
    # se il modello fitta bene i dati. Questo mi dà la misura di come il modello 
    # fitta i dati, ma non è da intendere come previsione, quella si fa sul TEST
    A.loc['Train', 'MAE%'] = np.mean(np.abs(y_train - y_train_pred)) / np.mean(y_train)
    A.loc['Train', 'RMSE%'] = np.sqrt( np.mean((y_train - y_train_pred)**2) ) / np.mean(y_train)
    A.loc['Train', 'BIAS%'] = np.mean(y_train - y_train_pred) / np.mean(y_train)
    
    resid_train = y_train - y_train_pred
    A.loc['Train', 'R^2'] = 1 - resid_train.var()/y_train.var()
    
    
    # PERFORMANCE DI PREVISIONE 
    A.loc['Test', 'MAE%'] = np.mean(np.abs(y_test - y_pred)) / np.mean(y_test)
    A.loc['Test', 'RMSE%'] = np.sqrt( np.mean((y_test - y_pred)**2) ) / np.mean(y_test)
    A.loc['Test', 'BIAS%'] = np.mean(y_test - y_pred) / np.mean(y_test)
    
    resid_pred = y_test - y_pred
    A.loc['Test', 'R^2'] = 1 - resid_pred.var()/y_test.var()

    # Sistemo il dataframe
    A = A.astype(float).round(4)
    
    return(A)


####################################################################################################################################


# Riaggregare y_train e y_train_pred in un dataset => df_y_train & df_y_train_pred

def aggregate_TRAIN(array_TRAIN, A, xlen, ylen, test_loops, t):
    
    # array_TRAIN può essere un y_train_pred o un y_train che voglio portare da array numpy di nuovo a df
    # A è il dataframe da cui prendo il nome dei codici (righe) e dei periodi (colonne) nel ricreare i df
    # t è il periodo di forecast futura che vado a considerare per riaggregarlo da array a df
    
    # Calcolo quanti sono i loop che possono essere fatti a partire da un dataframe iniziale A
    loops = A.shape[1] - xlen - ylen + 1
    
    if test_loops > 0 : # Se faccio validazione (i.e. test_loops > 0)
        df_TRAIN = pd.DataFrame(array_TRAIN[:, t].reshape((loops-test_loops, -1)).T, 
                                index = A.index,
                                columns = A.columns[(xlen + t) : (-test_loops - ylen + t + 1)])
    
    else: # Se faccio semplice forecast futura (i.e.test_loops = 0)
        if (-ylen + t + 1) == 0:
            df_TRAIN = pd.DataFrame(array_TRAIN[:, t].reshape((loops-test_loops, -1)).T, 
                                    index = A.index,
                                    columns = A.columns[(xlen + t) : ])
        else:
            df_TRAIN = pd.DataFrame(array_TRAIN[:, t].reshape((loops-test_loops, -1)).T, 
                                    index = A.index,
                                    columns = A.columns[(xlen + t) : (- ylen + t + 1)])
    return(df_TRAIN)


# ========================================================================================================================


# Riaggregare TEST e PREVISIONE in dei dataset => df_y_test e df_y_pred

def aggregate_TEST(array_TEST, A, xlen, ylen, test_loops, t, how, future_cols):
    
    # array_TEST può essere un y_test o un y_pred che voglio portare da array numpy di nuovo a df
    # A è il dataframe da cui prendo il nome dei codici (righe) e dei periodi (colonne) nel ricreare i df
    # t è il periodo di forecast futura che vado a considerare per riaggregarlo da array a df
    
    # Calcolo quanti sono i loop che possono essere fatti a partire da un dataframe iniziale A
    loops = A.shape[1] - xlen - ylen + 1
    
    if test_loops > 0 : # Se faccio validazione (i.e. test_loops > 0)
        if (-ylen + t + 1) == 0:
            df_TEST = pd.DataFrame(array_TEST[:, t].reshape((test_loops, -1)).T, 
                                   index = A.index,
                                   # il +1 perché il secondo estremo non è incluso nello slicing di un array
                                   columns = A.columns[(-test_loops - ylen + t + 1):])
        else:
            df_TEST = pd.DataFrame(array_TEST[:, t].reshape((test_loops, -1)).T, 
                                   index = A.index,
                                   # il +1 perché il secondo estremo non è incluso nello slicing di un array
                                   columns = A.columns[(-test_loops - ylen + t + 1):(-ylen + t + 1)])
        return(df_TEST)
    
    else: # Se faccio semplice forecast futura (i.e.test_loops = 0)
        if how == 'test':
            print('No df_TEST perché non sono in validazione')
        if how == 'pred':
            df_TEST = pd.DataFrame(array_TEST, index = A.index, 
                                   columns = future_cols)
            return (df_TEST)
        

####################################################################################################################################


# Plottare risultati

def make_plot_leaves(A, df_y_train_pred, df_y_test, df_y_pred, lista_leaves, test_loops):
    
    for leaf in lista_leaves:
    
        fig, ax = plt.subplots(2, 1, figsize = [20, 15])
        
        # ------------------------------------------------------ GRAFICO ASSE 0
        
        # Grafico che mostra l'intera serie + le previsioni + il fitting del train
        ax[0].plot(A.loc[leaf, :], marker = 'o', markersize = 3, label = 'original (train & test)', 
                   linestyle = '--', lw = 2, c = 'black')
        ax[0].plot(df_y_pred.loc[leaf, :], marker = 'o', markersize = 3, label = 'prediction', lw = 4, c = 'violet')
        ax[0].plot(df_y_train_pred.loc[leaf, :], marker = 'o', markersize = 3, label = 'train_fitting', lw = 4, c = 'lime')
            
        
        
        # ------------------------------------------------------ GRAFICO ASSE 1
        if test_loops > 0:
            ax[1].plot(df_y_test.loc[leaf, :], marker = 'o', markersize = 5, label = 'original (test)', linestyle = '--', lw = 2, c = 'black') 
        
        
        ax[1].plot(df_y_pred.loc[leaf, :], marker = 'o', label = 'prediction', lw = 4, c = 'violet')
            
        
        # Disegno la griglia ascisse dell'intera serie su ax[0]
        step = 3
        ax[0].set_xticks( A.T.index[ [j for j in np.arange(0, len(A.T.index), step)] ] )
        ax[0].tick_params(axis='x', rotation = 45) 
        
        # Disegno la griglia ascisse dell'intera serie su ax[1]
        step = 1
        ax[1].set_xticks( df_y_pred.T.index[ [j for j in np.arange(0, len(df_y_pred.T.index), step)] ] )
        ax[1].tick_params(axis='x', rotation = 45)
            
        # Legend
        ax[0].legend(edgecolor = 'red', facecolor = 'white', fontsize = 13, loc = 'best')
        ax[1].legend(edgecolor = 'red', facecolor = 'white', fontsize = 13, loc = 'best')
        
        fig.suptitle(leaf, fontsize = 15)
            
    plt.show()