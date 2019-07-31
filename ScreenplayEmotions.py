import os
# Analysis
import pandas as pd
import numpy as np
import scipy as sp
# Plotting
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
# NLP packages
from nltk.tokenize.casual import casual_tokenize
from py_lex import EmoLex
# Progress indicator
# from tqdm import tqdm

matplotlib.rcParams['figure.figsize'] = (20, 15)
sns.set_style('whitegrid')

class MovieCollection():
    def __init__(self, metadata_path, screenplay_dir, lexicon_path):
        self.metadata_path = metadata_path
        self.screenplay_dir = screenplay_dir
        self.lexicon_path = lexicon_path
        # self.load_metadata(metadata_path, screenplay_dir)
        self.lex = Lexicon(lexicon_path)
        print("Use read_emotions(N, M, D, G) to get started, load_emotions(config_file) if you have previous calculations or look at 'failed_titles' to troubleshoot.")
    
    def read_emotions(self, N, M, D, G):
        # (N)umber of sections, (M)emory, (D)ecay, (G)enre weight
        self.N, self.M, self.D, self.G = (N, M, D, G)
        print("Loading emotions ...")
        self.timeline_collection = self.get_timeline_collection()
        print("Converting emotions to long form ...")
        self.timeline_collection_long = self.get_long_timelines()
        self.set_genre_weight(G)
        print("Done.")
        
    def save_emotions(self, filename='screenplay-emotions.cfg'):
        data = ",".join(
            map(str,
                [self.N, self.M, self.D, self.G,
                 'timeline_long.csv']
               )
        )
        with open(filename, 'w') as f:
            f.write(data)
        # self.timeline_collection.to_csv('timeline.csv')
        self.timeline_collection_long.to_csv('timeline_long.csv', float_format='%.8f')
        print("Saved to %s" % filename)
                
    def load_emotions(self, filename='screenplay-emotions.cfg'):
        with open(filename, 'r') as f:
            config = f.read()
        config = config.split(',')
        self.N, self.M = map(int, config[:2])
        self.D, self.G = map(float, config[2:4])
        # timeline_collection_path = config[4]
        timeline_collection_long_path = config[4]
        self.timeline_collection_long = pd.read_csv(timeline_collection_long_path, index_col=0)
        self.get_short_timelines_from_long()
        self.set_genre_weight(self.G)
        print("Done.")
    
    def set_genre_weight(self, G):
        print("Appending genres to emotions ...")
        self.build_genre_matrix(G)
        self.timeline_collection_long_genre = pd.concat(
            [self.genre_binary, self.timeline_collection_long], 1)
        print("Calculating distance matrix ...")
        self.distance_between_movies = self.distance_matrix()
    
    def get_distance_to(self, title, n=None):
        if n is not None:
            return self.distance_between_movies[title].sort_values()[:n]
        else:
            return self.distance_between_movies[title].sort_values()
    
    def highlight(self, series, f):  # Pandas dataframe highlight
        selection = series == f([s for s in series if (s < 1 and s > 0)])
        return ['color: red' if value else '' for value in selection]
    
    def percent(self, v, precision='0.2'):  # Pandas dataframe percent
        try:
            return "{{:{}%}}".format(precision).format(v)
        except:
            raise TypeError("Numeric type required")
        
    def similarity(self, titles=[], highlight=None):
        comparisons = pd.DataFrame(columns = titles)
        for A in titles:
            row = {}
            for B in titles:
                # Distance is the opposite of similarity
                # Similarity = 1 - distance
                row[B] = 1 - self.get_distance_to(A)[B]
            comparisons = comparisons.append(pd.DataFrame(row, columns = titles, index=[A]))
        # total = comparisons.sum(axis=1)
        # comparisons['Total'] = total
        if highlight is not None:
            comparisons = comparisons.style.apply(lambda series: self.highlight(series, highlight), axis=1)
        return comparisons
    
    def mean_emotions(self, titles=[]):
        # Movies in Columns, emotions in rows
        df = pd.DataFrame()
        for title in titles:
            ix = self.titles.index(title)
            df[title] = self.timeline_collection[ix].mean(axis=0)
        return df
    
    def relative_emotions(self, titles=[]):
        # Movies in Columns, emotions in rows
        df = pd.DataFrame()
        for title in titles:
            ix = self.titles.index(title)
            df[title] = self.timeline_collection[ix].sum(axis=0)
            df[title] = df[title]/sum(df[title])
            df[title].apply(lambda x: self.percent(x, '0.1'))
        return df
    
    def emotion_diff(self, titles, kind):
        df = self.timeline_collection_long[self.timeline_collection_long.index.isin(titles)]
        dft = df.transpose()
        # Normalize within movie
        dft = (dft - dft.mean()) / (dft.max() - dft.min())
        dft = dft.transpose()
        # Movies in cols, emotions in rows
        # for i, e in enumerate(self.lex.emotions):
        
        ##### specify if diff by scene or diff by total
            
        
        
    def emotions_plot(self, titles=[], include=[], exclude=['positive', 'negative'], top=4, fill=True):
        ixs = [self.titles.index(i) for i in titles]
        dfs = [self.timeline_collection[i] for i in ixs]
        return self.lex.plot_sxs(dfs, titles, include, exclude, top, fill)

    def emotions_barplot(self, titles=[], include=[], exclude=['positive', 'negative'], top=4, fill=True):
        ixs = [self.titles.index(i) for i in titles]
        dfs = [self.timeline_collection[i] for i in ixs]
        return self.lex.barplot_sxs(dfs, titles, include, exclude, top, fill)


    # Create a list of dataframes:
    # Scenes (rows) vs emotions (columns) for each movie
    def get_timeline_collection(self):
        N, M, D = self.N, self.M, self.D
        length = len(self.filenames)
        emotion_timelines = [pd.DataFrame]*len(self.filenames)
        for i, fn in enumerate(self.filenames):
            pct = str(round(((i+1)/length)*100, 2))+'%'
            print('Processing \"%s\" ... (%s)'% (self.filenames[i], pct))
            with open(os.path.join(self.screenplay_dir, fn), 'r') as f:
                s = f.read()
            emotions = self.lex.emotions_with_decay(s, N, M, D)
            # Account for empty screenplays
            emotion_timelines[i] = emotions
        return emotion_timelines
    
    # Build a single large df:
    # Movies in rows, emotion in scene J in columns.
    def get_long_timelines(self):
        N = self.N
        scenes = list(range(N))
        cols = self.expandGrid(self.lex.emotions, scenes)
        length = len(self.timeline_collection)
        emo_dist = pd.DataFrame(columns=[''.join([emo, str(scene)]) for emo, scene in cols])
        for i, movie in enumerate(self.timeline_collection):
            pct = str(round(((i+1)/length)*100, 2))+'%'
            print('Processing \"%s\" ... (%s)'% (self.titles[i], pct))
            row = dict()
            for scene in scenes:
                scene_emos = dict()
                for emo in self.lex.emotions:
                    scene_emos[emo + str(scene)] = movie[emo][scene]
                row.update(scene_emos)
            emo_dist = emo_dist.append(pd.DataFrame(row, index=[self.titles[i]]))
        return emo_dist
    
    def get_short_timelines_from_long(self):
        N = self.N
        scenes = range(N)
        self.timeline_collection = []
        n_movies = self.timeline_collection_long.shape[0]
        for m in range(n_movies):
            scene_list = [{}]*N
            for r in scenes:
                row = {}
                for emo in self.lex.emotions:
                    row[emo] = self.timeline_collection_long[emo+str(r)][m]
                scene_list[r] = row
            df = pd.DataFrame(scene_list)
            self.timeline_collection.append(df)
    
    def expandGrid(self, K, L):
        return [(k, l) for k in K for l in L]

    def distance_matrix(self):
        # Cosine distance of emotions and scenes between movies
        emo_dist_t = self.timeline_collection_long_genre.transpose()
        # Normalize within movie
        emo_norm = (emo_dist_t - emo_dist_t.mean()) / (emo_dist_t.max() - emo_dist_t.min())
        emo_norm = emo_norm.transpose()
        distmat = sp.spatial.distance.pdist(emo_norm, 'cosine')
        distmat = sp.spatial.distance.squareform(distmat)
        distmat = pd.DataFrame(distmat, columns = emo_norm.index, index = emo_norm.index)
        return distmat    
        
    def load_metadata(self, metadata_path, screenplay_dir):
        titles = []
        genres = []
        # urls = []
        filenames = []

        with open(metadata_path, 'r') as f:
            rows = f.read()
        rows = rows.split('\n')

        for r in range(len(rows)):
            try:
                title, genre, filename = rows[r].split('\t')
            except:
                print("Error unpacking line %s:" % r, repr(rows[r].split('\t')))
                return False
            titles.append(title)
            genres.append(genre.split(','))
            filenames.append(filename)
        
        # Extract screenplay filenames from URLs
        # filenames = [u.replace('/scripts/', '').replace('.html', '') for u in urls]
        # for i, fn in enumerate(filenames):
        #     filenames[i] = ''.join([char for char in fn if char.isalnum()])+'.txt'

        pop_indexes = []
        failed_titles = []
        
        # Validate and correct metadata.
        for i, fn in enumerate(filenames):
            try:
                with open(os.path.join(screenplay_dir, fn), 'r') as f:
                    contents = f.read()
                    # Ignore if less than 1000 words
                    if len(contents.split(' ')) < 1000:
                        pop_indexes.append(i)
                        failed_titles.append(titles[i])
                    pass
            except:
                pop_indexes.append(i)
                failed_titles.append((titles[i],filenames[i]))

        # Get rid of failed items
        for i in reversed(pop_indexes):
            titles.pop(i)
            genres.pop(i)
            # urls.pop(i)
            filenames.pop(i)

        print("Successfully loaded %d items. Failed to load %d items." % (
                len(filenames), len(pop_indexes))
             )
        
        metadata = (titles, genres, filenames, failed_titles)
        self.titles, self.genres, self.filenames, self.failed_titles = metadata
        return metadata
    
    def build_genre_matrix(self, weight=.3):
        genre_types = []
        for i, m in enumerate(self.genres):
            for j, g in enumerate(m):
                if g not in genre_types:
                    genre_types.append(g)
        genre_types, len(genre_types)
        genre_binary = pd.DataFrame(columns = genre_types)
        for i, m in enumerate(self.genres):
            gs = [weight if g in m else 0 for g in genre_types]
            d = pd.DataFrame(dict(zip(genre_types, gs)), index=[self.titles[i]])
            genre_binary = genre_binary.append(d)
        self.genre_binary = genre_binary
        return genre_binary
    
    

    
    
        
class Lexicon():
    # Expects use of Plutchik's eight emotions
    # in the lexicon + 2 sentiments
    def __init__(self, lexicon_path):
        self.lexicon = EmoLex(lexicon_path)
        self.emotions = ['anger',
                        'anticipation', 
                        'disgust', 
                        'fear', 
                        'joy', 
                        'sadness', 
                        'surprise', 
                        'trust',
                        'positive',
                        'negative']
        self.chart_colors = {'sadness':'b',
                        'trust':'g',
                        'anger':'r',
                        'fear':'#a52a2a',
                        'anticipation':'#ff9900',
                        'joy':'c',
                        'surprise':'#dd22dd',
                        'disgust':'#550055',
                        'negative':'k',
                        'positive':'#aaaaaa'}
        self.chart_colors = {'sadness':'b',
                        'trust':'g',
                        'anger':'r',
                        'fear':'#a52a2a',
                        'anticipation':'#ff9900',
                        'joy':'c',
                        'surprise':'#dd22dd',
                        'disgust':'#550055',
                        'negative':'k',
                        'positive':'#aaaaaa'}
        
        
    def do_sentiments(self, tokens):
        anno = self.lexicon.annotate_doc(tokens)
        summ = self.lexicon.summarize_annotation(doc=tokens, annotation=anno)
        return summ
    
    def emotions_with_decay(self, s, N, M, D):
        # (s)tring, (N)umber of slices
        # slices in (M)emory, (D)ecay factor
        tokens = casual_tokenize(s)
        len_tokens = len(tokens)
        base_window = dict(zip(self.emotions, [0]*len(self.emotions)))
        summary = [base_window]*N
        priors = [base_window]*M
        carry = 0
        done = 0
        for i in range(0, N):
            w_size = int((len_tokens+carry)/N)
            carry = w_size - ((len_tokens+carry)/N)
            w = tokens[done:min(done+w_size, len_tokens)]
            done += w_size
            summ = self.do_sentiments(w)
            # Add decay values to summary
            summ_mem = summ.copy()
            for j in range(M):
                summ_mem = self.add_dict(summ_mem, priors[j])
            summary[i] = summ_mem
            # shift prior windows
            priors.pop(0)
            priors.append(summ)
            # decay prior windows
            priors = self.decay(priors, D)
        return pd.DataFrame(summary)

    def decay(self, decay_list, factor):
        for i, d in enumerate(decay_list):
            decay_list[i] = dict([(key, d[key]*factor) for key in d.keys()])
        return decay_list
        
    def add_dict(self, A, B):
        keys = A.keys()
        new_dict = dict([(key, A[key]+B[key]) for key in keys])
        return new_dict
        
    def _top_in_dict(self, emo_dict, podium_size):
        # Expects a dict with emotions:values for a single scene
        # Returns dict with same keys, and all values are zero
        # except for the podium values.
        podium = {}
        sorted_keys = sorted(emo_dict, key=emo_dict.get, reverse=True)
        for k in range(len(sorted_keys)):
            key = sorted_keys[k]
            if k < podium_size:
                podium[key] = emo_dict[key]
            else:
                podium[key] = 0
        return podium

    def top_emotions_total(self, df, topx):
        totals = {}
        emos = df.variable.unique()
        for emo in emos:
            totals[emo] = sum(df.value[df.variable == emo])
        top_emos = self._top_in_dict(totals, topx)
        return top_emos
    
    def plot_emotions(self, df, title, include, exclude, top, fill=True, ax=None):
        h_melt = df.copy()
        h_melt["scene"] = list(range(h_melt.shape[0]))
        h_melt = pd.melt(h_melt, id_vars=["scene"])
        if include != []:
            h_melt = h_melt[h_melt.variable.isin(include)]
        if exclude != []:
            h_melt = h_melt[~h_melt.variable.isin(exclude)]
        top_emos = self.top_emotions_total(h_melt, top)
        use_emos = [emo for emo in sorted(top_emos, key=top_emos.get, reverse=True) if top_emos[emo] != 0]
        if ax is None:
            fig, ax = plt.subplots(1)
        length = df.shape[0]
        for i, emo in enumerate(use_emos):
            p = ax.plot(range(1,length+1), df[emo], linewidth=4, color=self.chart_colors[emo], alpha=.5, label=emo)
            if fill: p = ax.fill_between(range(1,length+1), df[emo], color=self.chart_colors[emo], alpha=.05)
        p = ax.set_title(title, fontsize=16, fontweight='bold')
        p = ax.set_xlabel("Section", fontsize=14)
        p = ax.set_ylabel("Fraction of words", fontsize=14)
        handles, labels = ax.get_legend_handles_labels()
        p = ax.legend(handles, labels, loc='upper left', fancybox=True, framealpha=.9, shadow=True, fontsize=14)
        p = ax.set_autoscalex_on(False)
        p = ax.set_xlim([1, length])
        return ax

    def plot_sxs(self, dfs, titles, include, exclude, top, fill):
        if len(titles) == 1:
            fig, axs = plt.subplots(len(titles), sharex = True)
            axs = [axs]
        else:
            fig, axs = plt.subplots(len(titles), sharex = True)
        for i in range(len(axs)):
            axs[i] = self.plot_emotions(dfs[i], titles[i], include, exclude, top, fill, axs[i])
        # return axs

    def barplot_sxs(self, dfs, titles, include, exclude, top, fill):
        if len(titles) == 1:
            fig, axs = plt.subplots(len(titles), sharex = True)
            axs = [axs]
        else:
            fig, axs = plt.subplots(len(titles), sharex = True)
        for i in range(len(axs)):
            axs[i] = self.barplot_emotions(dfs[i], titles[i], include, exclude, top, fill, axs[i])
        # return axs

    def barplot_emotions(self, df, title, include, exclude, top, fill=True, ax=None):
        h_melt = df.copy()
        h_melt["scene"] = list(range(h_melt.shape[0]))
        h_melt = pd.melt(h_melt, id_vars=["scene"])
        if include != []:
            h_melt = h_melt[h_melt.variable.isin(include)]
        if exclude != []:
            h_melt = h_melt[~h_melt.variable.isin(exclude)]
        top_emos = self.top_emotions_total(h_melt, top)
        use_emos = [emo for emo in sorted(top_emos, key=top_emos.get, reverse=True) if top_emos[emo] != 0]
        if ax is None:
            fig, ax = plt.subplots(1)
        length = df.shape[0]
        for i, emo in enumerate(use_emos):
            p = ax.bar(range(1,length+1), df[emo], linewidth=4, color=self.chart_colors[emo], alpha=.5, label=emo)
            # if fill: p = ax.fill_between(range(1,length+1), df[emo], color=self.chart_colors[emo], alpha=.05)
        p = ax.set_title(title, fontsize=16, fontweight='bold')
        p = ax.set_xlabel("Section", fontsize=14)
        p = ax.set_ylabel("Fraction of words", fontsize=14)
        handles, labels = ax.get_legend_handles_labels()
        p = ax.legend(handles, labels, loc='upper left', fancybox=True, framealpha=.9, shadow=True, fontsize=14)
        p = ax.set_autoscalex_on(False)
        p = ax.set_xlim([1, length])
        return ax
    

    

        