# chess_bot_2
Simple machine learning model that learns chess by playing itself

## purpose
Experimenting with different methods to have a bot learn chess given legal moves to practice more advanced Python concepts

## current version
Runs a number of simulated chess games against itself, utilizing various functions in order to improve on random choices from legal moves. E.g prioritizing captures, piece development, control of the center, and utilization of piece matricies heatmaps. Later versions will use ML to fine tune variables these functions use.

Current version mostly simulates draws (80%+) due to stalemate because of extensively long games. This is because both sides are utilizing the same strategies. Plan to implement factors to reward more aggressive plays later.
