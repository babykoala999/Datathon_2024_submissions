from urllib import request
from flask import Flask, render_template, jsonify
from predict import predictWinner
app = Flask(__name__, template_folder='template')

@app.route('/')
def index():
    return render_template('sportspart2.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Parse request parameters
    data = request.json
    home_team = data.get('home_team')
    away_team = data.get('away_team')
    toss_winner = data.get('toss_winner')
    toss_decision = data.get('toss_decision')
    venue = data.get('venue')

    winner_prediction = predictWinner(home_team, away_team, toss_winner, toss_decision, venue)
    return jsonify({'predicted_winner': winner_prediction})


if __name__ == '__main__':
    app.run(debug=True)

'''@app.route('/predict')
def result():
    return render_template('result.html')
'''
