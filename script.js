// Game state
let gameActive = true;

// Move history for the current game: array of { board: number[], moveIdx: number, player: string }
let moveHistory = [];

// ── Neural Network Setup ───────────────────────────────────────────────────
// One model shared for both X and O perspectives.
// Board encoding: from the perspective of the current player, own pieces = 1, opponent = -1, empty = 0.
// We always encode the board from the CPU's (O) perspective when training CPU moves,
// and from X's perspective when training X moves.

const model = tf.sequential();

// Hidden layers with more capacity for better generalisation
model.add(tf.layers.dense({ units: 36, inputShape: [9], activation: 'relu' }));
model.add(tf.layers.dense({ units: 36, activation: 'relu' }));

// Output: a preference score for each of the 9 cells
model.add(tf.layers.dense({ units: 9, activation: 'linear' }));

model.compile({ optimizer: tf.train.adam(0.001), loss: 'meanSquaredError' });

// ── Board Encoding ─────────────────────────────────────────────────────────

// Returns a 9-element array: X→1, O→-1, empty→0 (absolute encoding)
function prepareBoard() {
    const cells = Array.from(document.querySelectorAll('.cell'));
    return cells.map(cell =>
        cell.innerHTML === 'X' ? 1 : (cell.innerHTML === 'O' ? -1 : 0)
    );
}

// Returns the board flipped to the perspective of `player`:
// own pieces = 1, opponent pieces = -1, empty = 0.
function boardFromPerspective(board, player) {
    const sign = player === 'X' ? 1 : -1;
    return board.map(v => v * sign);
}

// ── UI helpers ─────────────────────────────────────────────────────────────

function setStatus(message) {
    document.getElementById('status').textContent = message;
}

// ── CPU Move ──────────────────────────────────────────────────────────────

async function cpuMove() {
    const cells = Array.from(document.querySelectorAll('.cell'));
    const board = prepareBoard();
    const emptyCells = board.map((v, i) => v === 0 ? i : -1).filter(i => i >= 0);

    if (emptyCells.length === 0) return;

    // Encode board from O's perspective (own=1, opponent=-1)
    const boardO = boardFromPerspective(board, 'O');
    const inputTensor = tf.tensor2d([boardO]);
    const prediction = model.predict(inputTensor);
    const values = await prediction.data();
    inputTensor.dispose();
    prediction.dispose();

    // Choose the empty cell with the highest predicted score
    let bestIdx = emptyCells[0];
    let bestVal = -Infinity;
    for (const i of emptyCells) {
        if (values[i] > bestVal) {
            bestVal = values[i];
            bestIdx = i;
        }
    }

    // Record this move before placing it
    moveHistory.push({ board: boardO.slice(), moveIdx: bestIdx, player: 'O' });

    cells[bestIdx].innerHTML = 'O';
}

// ── Training ───────────────────────────────────────────────────────────────
// reward: +1 if the player whose moves we're training won, -1 if they lost, 0 for tie.
// We apply discounting so earlier moves in the game get a slightly smaller signal.

async function trainModel(winner) {
    // Nothing to train if there were no recorded moves
    if (moveHistory.length === 0) return;

    const GAMMA = 0.9; // discount factor

    // Separate X and O move histories and determine their rewards
    const xMoves = moveHistory.filter(m => m.player === 'X');
    const oMoves = moveHistory.filter(m => m.player === 'O');

    let xReward, oReward;
    if (winner === 'X') {
        xReward = 1;
        oReward = -1;
    } else if (winner === 'O') {
        xReward = -1;
        oReward = 1;
    } else {
        // Tie — small negative reward to encourage winning attempts
        xReward = -0.1;
        oReward = -0.1;
    }

    const trainingBoards = [];
    const trainingTargets = [];

    // Build training samples for all recorded moves
    for (const movesForPlayer of [xMoves, oMoves]) {
        const baseReward = movesForPlayer === xMoves ? xReward : oReward;

        for (let t = 0; t < movesForPlayer.length; t++) {
            const { board, moveIdx } = movesForPlayer[t];

            // Discount: moves earlier in the game get a weaker signal
            const stepsFromEnd = movesForPlayer.length - 1 - t;
            const discountedReward = baseReward * Math.pow(GAMMA, stepsFromEnd);

            // Get the model's current predictions for this board state so we
            // only update the target for the cell that was actually played.
            // This is the standard Q-learning / policy gradient approach.
            const inputTensor = tf.tensor2d([board]);
            const predTensor = model.predict(inputTensor);
            const pred = await predTensor.data();
            inputTensor.dispose();
            predTensor.dispose();

            const target = Array.from(pred);
            target[moveIdx] = discountedReward;

            trainingBoards.push(board);
            trainingTargets.push(target);
        }
    }

    if (trainingBoards.length === 0) return;

    const xTensor = tf.tensor2d(trainingBoards);
    const yTensor = tf.tensor2d(trainingTargets);

    await model.fit(xTensor, yTensor, { epochs: 5, shuffle: true, verbose: 0 });

    xTensor.dispose();
    yTensor.dispose();
}

// ── Win / Tie Detection ────────────────────────────────────────────────────

function checkWinner() {
    const cells = Array.from(document.querySelectorAll('.cell'));
    const combos = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        [0, 4, 8], [2, 4, 6],
    ];

    for (const [a, b, c] of combos) {
        const v = cells[a].innerHTML;
        if (v && v === cells[b].innerHTML && v === cells[c].innerHTML) {
            return v; // 'X' or 'O'
        }
    }

    if (cells.every(cell => cell.innerHTML !== '')) {
        return 'tie';
    }

    return null; // game still going
}

// ── Board Reset ────────────────────────────────────────────────────────────

function clearBoard() {
    document.querySelectorAll('.cell').forEach(cell => { cell.innerHTML = ''; });
    moveHistory = [];
    gameActive = true;
    setStatus("Your turn (X)");
}

// ── Main Click Handler ─────────────────────────────────────────────────────

function handleCellClick(cell) {
    cell.addEventListener('click', async function () {
        if (this.innerHTML !== '' || !gameActive) return;

        // --- Human plays X ---
        const boardBeforeX = boardFromPerspective(prepareBoard(), 'X');
        const xIdx = Array.from(document.querySelectorAll('.cell')).indexOf(this);
        moveHistory.push({ board: boardBeforeX, moveIdx: xIdx, player: 'X' });

        this.innerHTML = 'X';
        setStatus("CPU is thinking...");

        // Disable further clicks while CPU thinks
        gameActive = false;

        const afterX = checkWinner();
        if (afterX) {
            await endGame(afterX);
            return;
        }

        // --- CPU plays O ---
        await cpuMove();

        const afterO = checkWinner();
        if (afterO) {
            await endGame(afterO);
            return;
        }

        // Game continues
        gameActive = true;
        setStatus("Your turn (X)");
    });
}

async function endGame(result) {
    gameActive = false;

    if (result === 'X') {
        setStatus("You win!");
    } else if (result === 'O') {
        setStatus("CPU wins!");
    } else {
        setStatus("It's a tie!");
    }

    // Train on the completed game
    await trainModel(result);

    document.getElementById('newGameButton').style.display = 'block';
}

// ── Initialization ─────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', function () {
    const cells = document.querySelectorAll('.cell');
    cells.forEach(handleCellClick);

    setStatus("Your turn (X)");

    const newGameButton = document.getElementById('newGameButton');
    newGameButton.addEventListener('click', function () {
        clearBoard();
        newGameButton.style.display = 'none';
    });
});
