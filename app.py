from flask import Flask,request,render_template
import pandas as pd
import numpy as np
from nltk import sent_tokenize, word_tokenize, FreqDist
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.io as pio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import optuna
from optuna.samplers import TPESampler
import os,re
from keras.api.models import load_model
import google.generativeai as genai


app=Flask(__name__)

#AIzaSyD2Uir3fp_lS0SrW7n1Paqx4glsk-VHHBQ

def convert_paragraph_to_points(paragraph, num_points=15):
    sentences = sent_tokenize(paragraph)
    words = word_tokenize(paragraph.lower())
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    freq_dist = FreqDist(filtered_words)
    sentence_scores = {}
    for sentence in sentences:
        sentence_word_tokens = word_tokenize(sentence.lower())
        sentence_word_tokens = [word for word in sentence_word_tokens if word.isalnum()]
        score = sum(freq_dist.get(word, 0) for word in sentence_word_tokens)
        sentence_scores[sentence] = score
    sorted_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    key_points = sorted_sentences[:num_points]
    return key_points

def clean_text(text):
    return re.sub(r'\*\*|\*', '', text)

@app.route('/',methods=['GET','POST'])
def index():
    return render_template('index.html')

@app.route('/visualizations',methods=['GET','POST'])
def visualization():
    return render_template("visualization.html")

@app.route('/inputs',methods=['GET','POST'])
def inputs():
    return render_template('inputs.html')

@app.route('/submit_data',methods=['GET','POST'])
def datas():
    if request.method=="POST":
        discharge_time = request.form.get('discharge_time')
        decrement_3_6_3_4v = request.form.get('decrement_3_6_3_4v')
        max_voltage = request.form.get('max_voltage')
        time_at_4_15v = request.form.get('time_at_4_15v')
        time_constant_current = request.form.get('time_constant_current')
        charging_time = request.form.get('charging_time')

        discharge_time = float(discharge_time)
        decrement_3_6_3_4v = float(decrement_3_6_3_4v)
        max_voltage = float(max_voltage)
        time_at_4_15v = float(time_at_4_15v)
        time_constant_current = float(time_constant_current)
        charging_time = float(charging_time)

        arr=np.array([discharge_time,decrement_3_6_3_4v,max_voltage,time_at_4_15v,time_constant_current,charging_time])

        arr = arr.reshape(1, 6)
        model = load_model('rlu.h5')
        prediction = model.predict(arr)

        genai.configure(api_key='AIzaSyB92wEjbNgfe2vW6apS0gRMP3Z4nVCwVOc')
        model = genai.GenerativeModel('gemini-1.5-flash')
        content = model.generate_content(f"In the context of prediction of Battery RUl the Discharge time of Battery in seconds is {discharge_time} and Maximum voltage discharge is {max_voltage} and time at 4.15v in seconds is {time_at_4_15v} and time constant current in seconds is {time_constant_current} and charging time in seconds is {charging_time} and for all this RUL predicted by my model is {prediction[0][0]} so what do you think about it..? make sure that your answer should be based on the values given not about the model.?")
        generated_text = content.text
        key_points = convert_paragraph_to_points(generated_text)
        key_points = [clean_text(item) for item in key_points]

        cntnt=model.generate_content("In the view of Discharge time of Battery in seconds, Maximum discharge of voltage and time constant current in seconds and charging time in seconds recommend me an electric 2 wheeler and 4 four wheeler with cost, location, mileage, engine capacity and technology mainly")
        generated_texts = cntnt.text
        key_pints = convert_paragraph_to_points(generated_texts)
        key_pints = [clean_text(item) for item in key_pints]
        return render_template('inputs.html', keys=key_points,rec_points=key_pints)



@app.route('/visualize_test', methods=['POST'])
def visualize_test():
    data = pd.read_csv('Battery_RUL.csv')

    for i in data.columns.values:
        if len(data[i].value_counts().values) < 10:
            print(data[i].value_counts())

    # Outlier removal using z-scores
    out = []
    for i in data.columns.values:
        data['z_scores'] = (data[i] - data[i].mean()) / data[i].std()
        outlier = np.abs(data['z_scores'] > 3).sum()
        if outlier > 3:
            out.append(i)

    print(len(data))
    thresh = 3
    for i in out:
        upper = data[i].mean() + thresh * data[i].std()
        lower = data[i].mean() - thresh * data[i].std()
        data = data[(data[i] > lower) & (data[i] < upper)]

    print(len(data))

    univariate1 = request.form.get('univariate1')
    univariate2 = request.form.get('univariate2')
    plot_type = request.form.get('univariate3')

    plot_filename = f"{plot_type}_{univariate1}_{univariate2}.png"
    plot_path = f'static/images/{plot_filename}'

    if plot_type == 'histogram':
        fig, ax = plt.subplots()
        data[univariate1].hist(ax=ax)
        ax.set_title('Histogram of ' + univariate1)
        fig.savefig(plot_path, format='png')
        plt.close(fig)

    elif plot_type == 'boxplot':
        fig = plt.figure()
        sns.boxplot(data=data, x=univariate1, y=univariate2)
        plt.title(f'Boxplot of {univariate1} vs {univariate2}')
        plt.savefig(plot_path, format='png')
        plt.close(fig)

    elif plot_type == 'density':
        fig = plt.figure()
        sns.kdeplot(data=data, x=univariate1, y=univariate2, fill=True)
        plt.title(f'Density Plot of {univariate1} vs {univariate2}')
        plt.savefig(plot_path, format='png')
        plt.close(fig)

    elif plot_type == 'violin':
        fig = plt.figure()
        sns.violinplot(data=data, x=univariate1, y=univariate2)
        plt.title(f'Violin Plot of {univariate1} vs {univariate2}')
        plt.savefig(plot_path, format='png')
        plt.close(fig)

    elif plot_type == 'bar':
        fig = plt.figure()
        data[univariate1].value_counts().plot(kind='bar')
        plt.title(f'Bar Chart of {univariate1}')
        plt.savefig(plot_path, format='png')
        plt.close(fig)

    elif plot_type == 'scatter':
        fig = px.scatter(data, x=univariate1, y=univariate2,
                         title='Scatter Plot of {} vs {}'.format(univariate1, univariate2))
        pio.write_image(fig, plot_path, format='png')

    elif plot_type == 'heatmap':
        corr = data.corr()
        fig = plt.figure()
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title('Heatmap of Correlation Matrix')
        plt.savefig(plot_path, format='png')
        plt.close(fig)

    elif plot_type == 'contour':
        x = data[univariate1]
        y = data[univariate2]
        x = x[~x.isna()]
        y = y[~y.isna()]
        X, Y = np.meshgrid(np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100))
        Z = np.exp(-(X ** 2 + Y ** 2))
        fig = plt.figure()
        plt.contourf(X, Y, Z, cmap='viridis')
        plt.title('Contour Plot')
        plt.savefig(plot_path, format='png')
        plt.close(fig)

    elif plot_type == 'hexbin':
        fig, ax = plt.subplots()
        hb = ax.hexbin(data[univariate1], data[univariate2], gridsize=50, cmap='inferno')
        plt.colorbar(hb, ax=ax)
        ax.set_title('Hexbin Plot of {} vs {}'.format(univariate1, univariate2))
        plt.savefig(plot_path, format='png')
        plt.close(fig)


    elif plot_type == 'qq-plot':
        fig = plt.figure()
        sm.qqplot(data[univariate1], line='45')
        plt.title(f'Q-Q Plot of {univariate1}')
        plt.savefig(plot_path, format='png')
        plt.close(fig)


    elif plot_type == 'violin-box':
        fig = plt.figure()
        sns.violinplot(data=data, x=univariate1, y=univariate2)
        sns.boxplot(data=data, x=univariate1, y=univariate2, whis=np.inf)
        plt.title(f'Violin with Boxplot of {univariate1} vs {univariate2}')
        plt.savefig(plot_path, format='png')
        plt.close(fig)

    else:
        return "Invalid plot type", 400

    plot_url = f'/static/images/{plot_filename}'
    return render_template("visualization.html", plot_url=plot_url)


def summary_stats(column1, column2, stat_type):
    data = pd.read_csv('Battery_RUL.csv')
    print(data.columns)
    print(data.describe())
    print(data.info())
    print(data.isna().sum())

    # Detect columns with fewer than 10 unique values
    for i in data.columns.values:
        if len(data[i].value_counts().values) < 10:
            print(data[i].value_counts())

    # Outlier removal using z-scores
    out = []
    for i in data.columns.values:
        data['z_scores'] = (data[i] - data[i].mean()) / data[i].std()
        outlier = np.abs(data['z_scores'] > 3).sum()
        if outlier > 3:
            out.append(i)

    print(len(data))
    thresh = 3
    for i in out:
        upper = data[i].mean() + thresh * data[i].std()
        lower = data[i].mean() - thresh * data[i].std()
        data = data[(data[i] > lower) & (data[i] < upper)]

    print(len(data))

    result = ""

    if stat_type == "mean":
        result = f"Mean of {column1}: {data[column1].mean()}"
    elif stat_type == "median":
        result = f"Median of {column1}: {data[column1].median()}"
    elif stat_type == "mode":
        result = f"Mode of {column1}: {data[column1].mode()[0]}"
    elif stat_type == "variance":
        result = f"Variance of {column1}: {data[column1].var()}"
    elif stat_type == "std_dev":
        result = f"Standard Deviation of {column1}: {data[column1].std()}"
    elif stat_type == "kurtosis":
        result = f"Kurtosis of {column1}: {stats.kurtosis(data[column1].dropna())}"
    elif stat_type == "skewness":
        result = f"Skewness of {column1}: {stats.skew(data[column1].dropna())}"
    elif stat_type == "range":
        result = f"Range of {column1}: {data[column1].max() - data[column1].min()}"
    elif stat_type == "iqr":
        result = f"Interquartile Range of {column1}: {stats.iqr(data[column1].dropna())}"
    elif stat_type == "t-test":
        t_stat, p_val = stats.ttest_ind(data[column1].dropna(), data[column2].dropna())
        result = f"T-Test: t-statistic={t_stat}, p-value={p_val}"
    elif stat_type == "chi-square":
        contingency = pd.crosstab(data[column1], data[column2])
        chi2_stat, p_val, dof, ex = stats.chi2_contingency(contingency)
        result = f"Chi-Square Test: chi2-statistic={chi2_stat}, p-value={p_val}, degrees of freedom={dof}"
    elif stat_type == "anova":
        groups = [data[column1][data[column2] == g] for g in data[column2].unique()]
        f_stat, p_val = stats.f_oneway(*groups)
        result = f"ANOVA: F-statistic={f_stat}, p-value={p_val}"
    elif stat_type == "mann-whitney":
        u_stat, p_val = stats.mannwhitneyu(data[column1].dropna(), data[column2].dropna())
        result = f"Mann-Whitney U Test: U-statistic={u_stat}, p-value={p_val}"
    elif stat_type == "wilcoxon":
        w_stat, p_val = stats.wilcoxon(data[column1].dropna(), data[column2].dropna())
        result = f"Wilcoxon Signed-Rank Test: W-statistic={w_stat}, p-value={p_val}"
    elif stat_type == "kruskal-wallis":
        groups = [data[column1][data[column2] == g] for g in data[column2].unique()]
        h_stat, p_val = stats.kruskal(*groups)
        result = f"Kruskal-Wallis Test: H-statistic={h_stat}, p-value={p_val}"
    elif stat_type == "pearson-correlation":
        corr, _ = stats.pearsonr(data[column1].dropna(), data[column2].dropna())
        result = f"Pearson Correlation: {corr}"
    elif stat_type == "spearman-correlation":
        corr, _ = stats.spearmanr(data[column1].dropna(), data[column2].dropna())
        result = f"Spearman Correlation: {corr}"
    elif stat_type == "fisher-exact":
        contingency = pd.crosstab(data[column1], data[column2])
        odds_ratio, p_val = stats.fisher_exact(contingency)
        result = f"Fisher's Exact Test: Odds Ratio={odds_ratio}, p-value={p_val}"
    elif stat_type == "z-test":
        mean = data[column1].mean()
        std_dev = data[column1].std()
        z_stat = (mean - 0) / (std_dev / np.sqrt(len(data[column1])))
        p_val = stats.norm.sf(abs(z_stat)) * 2
        result = f"Z-Test: z-statistic={z_stat}, p-value={p_val}"
    elif stat_type == "lin-regression":
        from sklearn.linear_model import LinearRegression
        X = data[[column1]]
        y = data[column2]
        model = LinearRegression().fit(X, y)
        result = f"Linear Regression: Coefficient={model.coef_[0]}, Intercept={model.intercept_}"
    elif stat_type == 'OLS':
        import statsmodels.api as sm
        x = data[[column1]]
        y = data[[column2]]
        model = sm.OLS(y, x).fit()
        result = model.summary().as_html()
    return result


@app.route('/stat_test', methods=['POST'])
def stat_test():
    stats1 = request.form.get('stats1')
    stats2 = request.form.get('stats2')
    stats3 = request.form.get('stats3')

    if stats3:
        summary = summary_stats(stats1, stats2, stats3)
    else:
        summary = "Please select a summary statistic or test."
    return render_template('visualization.html', summary=summary)


@app.route('/optuna', methods=['GET', 'POST'])
def casuality_analysis():
    return render_template("casual.html")


@app.route('/c_analysis', methods=['GET', 'POST'])
def c_analysis():
    data = pd.read_csv('Battery_RUL.csv')

    out = []
    for i in data.columns.values:
        data['z_scores'] = (data[i] - data[i].mean()) / data[i].std()
        outlier = np.abs(data['z_scores'] > 3).sum()
        if outlier > 3:
            out.append(i)

    thresh = 3
    for i in out:
        upper = data[i].mean() + thresh * data[i].std()
        lower = data[i].mean() - thresh * data[i].std()
        data = data[(data[i] > lower) & (data[i] < upper)]

    corr = data.corr()['RUL']
    corr = corr.drop(['RUL', 'z_scores'])
    x_cols = [i for i in corr.index if corr[i] > 0]
    x = data[x_cols]
    y = data['RUL']

    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 100)
        max_depth = trial.suggest_int('max_depth', 2, 10)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=0
        )
        score = cross_val_score(model, x, y, n_jobs=-1, cv=3)
        accuracy = score.mean()
        return accuracy

    sampler = TPESampler(seed=10)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=2)

    best_params = study.best_params
    model = RandomForestClassifier(**best_params, random_state=0)
    model.fit(x, y)

    feature_importance = model.feature_importances_
    features_importance_zipped = zip(x.columns.values, feature_importance.tolist())


    fig_dir = os.path.join('static', 'images')
    os.makedirs(fig_dir, exist_ok=True)

    return render_template(
        "c_analysis.html",
        best_params=best_params,
        features_importance_zipped=features_importance_zipped,
    )
if __name__=='__main__':
    app.run()