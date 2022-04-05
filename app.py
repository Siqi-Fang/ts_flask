from flask import Flask, render_template, request
from memory_profiler import profile  # dont put in requirements

import model

app = Flask(__name__)
app.debug = True


@app.route('/', methods=['POST', 'GET'])
@profile
def home():  # put application's code here
    if request.method == 'POST':
        context = {}
        # read input and preprocess it
        tweet = request.form["tweet"]
        # make prediction
        sentiment, confidence = model.single_prediction(tweet)
        # map to styled html span
        SENT_MAP = {0: '<span class="text-white bg-danger">Negative</span>',
                    4: '<span class="text-white bg-success">Positive</span>'}
        # plot word importance & highlight text
        graph = model.plot_explanation(tweet)[0]
        tweet = model.plot_explanation(tweet)[1]
        sentiment = SENT_MAP[sentiment]

        context['graph'] = graph
        context['sentiment'] = sentiment
        context['confidence'] = "%.2f" % (confidence * 100)
        context['text'] = tweet

        return render_template('result.html', **context)  # ** drops key in htmls

    return render_template('home.html')


if __name__ == '__main__':
    app.run()
