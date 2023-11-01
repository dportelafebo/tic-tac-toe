// Initialize the current player as 'X'
let currentPlayer = "X";
let gameActive = true;

let X_train = [];  // to store board states
let y_train = [];  // to store best moves or outcomes

// Initialize the neural network
// Create a sequential model. A sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.
const model = tf.sequential();

// Add the hiden layer
// 'units' is the dimensionality of the output space
// 'inputShape' is the shape of the input. In this case it's a 1D array of length 9 (the tic tac toe board)
// 'activation' is the activation function to use. ReLU (Rectified Linear Unit) is a common choice.
model.add(tf.layers.dense({units: 9, inputShape: [9], activation:'relu'}))

// Add the output layer
// We use 'softmax' as the activation function. Softmax will make the outputs sum to 1, so they can be interpreted as probabilities
model.add(tf.layers.dense({units:9, activation:'softmax'}))

// Compile the model
// 'optimizer is the optimization algorithm to use. Adam is a good default choice
// 'loss' is the loss function to use. Mean Squared Error is a common choice for regression problems
model.compile({optimizer: 'adam', loss:'meanSquaredError'})

// Function to prepare board state for neural network
// This function converts the board state into a format that can be fed into the neural network.
function prepareBoard() {
    // Get all the cells as an array
    const cells = Array.from(document.querySelectorAll('.cell'))

    // Map each cell to a number: 'X' -> 1, 'O' -> -1, empty -> 0
    return cells.map(cell => cell.innerHTML === "X" ? 1 : (cell.innerHTML === "O" ? -1 : 0))
}


// Function to handle the click event on a cell
function handleCellClick(cell) {
    // Attach a click event listener to the cell
    cell.addEventListener("click", function () {
        // Only proceed if the cell is empty
        if (this.innerHTML === "" && gameActive) {
            // Place the current player's symbol ('X' or 'O') in the cell
            this.innerHTML = currentPlayer;
            if (checkForWin()) {
                // Set board unclickable
                gameActive = false;
                // Hide the New Game button again
                document.getElementById('newGameButton').style.display = 'none';
                // Show the New Game button
                document.getElementById('newGameButton').style.display = 'block';
                console.log(`${currentPlayer} won`);
            } else {
                cpuMove();  // Make the CPU move
                if (checkForWin()) {
                    gameActive = false;
                    document.getElementById('newGameButton').style.display = 'block';
                    console.log("CPU won");
                }
            }
        }
    });
}

// Function for CPU to make a move
async function cpuMove() {
    const cells = Array.from(document.querySelectorAll(".cell"));
    const emptyCells = cells.filter(cell => cell.innerHTML === "");

    if (emptyCells.length > 0) {
        const board = prepareBoard();

        // Use the model to predict the move probabilitites for the current board state
        const prediction = model.predict(tf.tensor2d([board]))

        // Extract probabilities from the tensor
        const values = await prediction.data()

        // Initialize variables to keep track of the best move
        let bestMoveIdx = 0;
        let bestMoveValue = -Infinity;

        // Loop through each move and its predicted value
        for (let i = 0; i < values.length; i++){
            // If the cell is empty and the move has a better value, update the best move
            if (board[i] === 0 && values[i] > bestMoveValue) {
                bestMoveValue = values[i];
                bestMoveIdx = i;
            }
        }

        // Make the best move
        cells[bestMoveIdx].innerHTML = "O"
    }
}

// Function to train the neural network based on game data
function trainModel() {
    // Dummy training data for the board state
    const X_train = [prepareBoard()];

    // Dummy training data for the move to make
    const y_train = Array(9).fill().map(() => Math.random()); // array of 9 random numbers

    // Train the model
    // 'tf.tensor2d' converts arrays into 2D tensors that TensorFlow.js can understand
    // 'epochs' is the number of times to go through the training data
    model.fit(tf.tensor2d(X_train), tf.tensor2d([y_train]), {epochs: 1})
}

// Function to clear the board
function clearBoard() {
    const cells = document.querySelectorAll(".cell");
    cells.forEach(cell => cell.innerHTML = "");
    gameActive = true; // allow the board to be clicked again
}

// Function to check for a win or a tie
function checkForWin() {
    // Get all cells
    const cells = Array.from(document.querySelectorAll(".cell"));

    // Define the winning combinations
    const winningCombination = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8], // Horizontal
        [0, 3, 6], [1, 4, 7], [2, 5, 8], // Vertical
        [0, 4, 8], [2, 4, 6]  // Diagonal
    ];

    // Check for a win
    for (let [a, b, c] of winningCombination) {
        if (cells[a].innerHTML && cells[a].innerHTML === cells[b].innerHTML && cells[a].innerHTML === cells[c].innerHTML) {
            trainModel(); // Train model after the game is over
            return true;
        }
    }

    // Check for a tie
    if (cells.every(cell => cell.innerHTML)) {
        console.log("Tie");
        trainModel(); // Train the model after the game is over
        return true;
    }
}

// Wait for the DOM to be fully loaded
document.addEventListener("DOMContentLoaded", function () {
    // Get all cells and attach click event handlers
    const cells = document.querySelectorAll(".cell");
    cells.forEach(handleCellClick);

    // Attach a click event listener to the "New Game" button
    const newGameButton = document.getElementById('newGameButton');
    newGameButton.addEventListener('click', function() {
    // Clear the board
    clearBoard();

    // Hide the "New Game" button
    newGameButton.style.display = 'none';
  });
});
