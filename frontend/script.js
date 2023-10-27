// Initialize the current player as 'X'
let currentPlayer = "X";

// Function to handle the click event on a cell
function handleCellClick(cell) {
    // Attach a click event listener to the cell
    cell.addEventListener("click", function () {
        // Only proceed if the cell is empty
        if (this.innerHTML === "") {
            // Place the current player's symbol ('X' or 'O') in the cell
            this.innerHTML = currentPlayer;
            if (checkForWin()) {
                // Hide the New Game button again
                document.getElementById('newGameButton').style.display = 'none';
                // Show the New Game button
                document.getElementById('newGameButton').style.display = 'block';
            }

            // Check for win or tie
            checkForWin();

            // Switch to the other player for the next turn
            currentPlayer = (currentPlayer === "X") ? "O" : "X";
        }
    });
}

// Function to clear the board
function clearBoard() {
    const cells = document.querySelectorAll(".cell");
    cells.forEach(cell => cell.innerHTML = "");
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
            console.log(`${currentPlayer} won`);
            return true;
        }
    }

    // Check for a tie
    if (cells.every(cell => cell.innerHTML)) {
        console.log("Tie");
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
