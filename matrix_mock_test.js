const fs = require("fs");
const chai = require("chai");
const sinon = require("sinon");
const sinonChai = require("sinon-chai");
const matrix = require("./matrix.js");
const {
    gauss,
    gauss_forward,
    gauss_backward,
    read_input,
} = require("./gauss_functions.js");

chai.use(sinonChai);
const expect = chai.expect;

function createMatrix(rows) {
    const m = new matrix(rows.length);
    for (let i = 0; i < rows.length; i++) {
        for (let j = 0; j < rows[i].length; j++) {
            m.set(i, j, rows[i][j]);
        }
    }
    return m;
}

describe("Mock методів класу matrix", function () {
    let sandbox;

    beforeEach(function () {
        sandbox = sinon.createSandbox();
    });

    afterEach(function () {
        sandbox.restore();
    });

    describe("get_matrix", function () {
        it("конструктор берёт матрицу из замоканного get_matrix", function () {
            const prepared = [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ];
            const mock = sandbox.mock(matrix.prototype);
            const getMatrix = mock
                .expects("get_matrix")
                .once()
                .withExactArgs(3)
                .returns(prepared);

            const m = new matrix(3);

            expect(m.matrix).to.equal(prepared);
            expect(getMatrix).to.have.been.calledOnce;
            expect(getMatrix).to.have.been.calledWithExactly(3);
            mock.verify();
        });
    });

    describe("set", function () {
        it("read_input записывает коэффициенты только через set", function () {
            const lines = fs.readFileSync("input.txt", "utf8").split("\n");
            lines.pop();
            const n = parseInt(lines[0], 10);
            const mock = sandbox.mock(matrix.prototype);
            const set = mock.expects("set").exactly(n * (n + 1));

            read_input();

            const first = lines[1].split(" ").map(Number);
            const last = lines[n].split(" ").map(Number);
            expect(set).to.have.been.calledWith(0, 0, first[0]);
            expect(set).to.have.been.calledWith(0, n, first[n]);
            expect(set).to.have.been.calledWith(n - 1, n, last[n]);
            mock.verify();
        });
    });

    describe("mull_add", function () {
        it("прямой ход вызывает mull_add с коэффициентом исключения", function () {
            const m = createMatrix([
                [2, 1, 5],
                [4, 3, 6],
            ]);
            const mock = sandbox.mock(m);
            // d = -(a[1][0] / a[0][0]) = -(4 / 2) = -2
            const mullAdd = mock.expects("mull_add").once().withExactArgs(1, 0, -2);
            mock.expects("swap_with_nonzero_row").never();

            gauss_forward(m);

            expect(mullAdd).to.have.been.calledOnce;
            expect(mullAdd).to.have.been.calledWithExactly(1, 0, -2);
            mock.verify();
        });

        it("прямой ход вызывает mull_add для каждого элемента под диагональю", function () {
            const m = createMatrix([
                [2, 1, 1, 1],
                [4, 1, 1, 1],
                [2, 2, 1, 1],
            ]);
            const mock = sandbox.mock(m);
            const first = mock.expects("mull_add").once().withExactArgs(1, 0, -2);
            const second = mock.expects("mull_add").once().withExactArgs(2, 0, -1);
            const third = mock.expects("mull_add").once().withExactArgs(2, 1, -2);

            gauss_forward(m);

            expect(first).to.have.been.calledWithExactly(1, 0, -2);
            expect(second).to.have.been.calledWithExactly(2, 0, -1);
            expect(third).to.have.been.calledWithExactly(2, 1, -2);
            mock.verify();
        });
    });

    describe("swap_with_nonzero_row", function () {
        it("прямой ход меняет строку, когда опорный элемент нулевой", function () {
            const m = createMatrix([
                [0, 1, 2],
                [3, 4, 5],
            ]);
            const mock = sandbox.mock(m);
            const swap = mock.expects("swap_with_nonzero_row").once().withExactArgs(0);
            mock.expects("mull_add").never();

            gauss_forward(m);

            expect(swap).to.have.been.calledOnce;
            expect(swap).to.have.been.calledWithExactly(0);
            mock.verify();
        });
    });

    describe("get", function () {
        it("обратный ход считает решение из замоканных элементов", function () {
            const m = new matrix(2);
            const mock = sandbox.mock(m);
            mock.expects("get_rows").once().returns(2);
            mock.expects("get_cols").once().returns(3);
            // i = 1: права частина 4, діагональ 2 → x1 = 2
            const rhs1 = mock.expects("get").withExactArgs(1, 2).once().returns(4);
            const diag1 = mock.expects("get").withExactArgs(1, 1).once().returns(2);
            // i = 0: a01 = 1, права частина 5, діагональ 2 → x0 = (5 - 1*2) / 2
            const coeff = mock.expects("get").withExactArgs(0, 1).once().returns(1);
            const rhs0 = mock.expects("get").withExactArgs(0, 2).once().returns(5);
            const diag0 = mock.expects("get").withExactArgs(0, 0).once().returns(2);

            const solution = gauss_backward(m);

            expect(solution).to.deep.equal([1.5, 2]);
            expect(rhs1).to.have.been.calledWithExactly(1, 2);
            expect(diag1).to.have.been.calledWithExactly(1, 1);
            expect(coeff).to.have.been.calledWithExactly(0, 1);
            expect(rhs0).to.have.been.calledWithExactly(0, 2);
            expect(diag0).to.have.been.calledWithExactly(0, 0);
            mock.verify();
        });
    });

    describe("exists_wrong_row і exists_zero_row", function () {
        function triangular() {
            return createMatrix([
                [2, 1, 5],
                [0, 2, 4],
            ]);
        }

        it("возвращает null и не проверяет нулевую строку, если есть противоречивая", function () {
            const m = triangular();
            const mock = sandbox.mock(m);
            const wrong = mock.expects("exists_wrong_row").once().returns(true);
            const zero = mock.expects("exists_zero_row").never();

            const solution = gauss(m);

            expect(solution).to.equal(null);
            expect(wrong).to.have.been.calledOnce;
            expect(zero).to.not.have.been.called;
            mock.verify();
        });

        it("возвращает null, если есть нулевая строка", function () {
            const m = triangular();
            const mock = sandbox.mock(m);
            const wrong = mock.expects("exists_wrong_row").once().returns(false);
            const zero = mock.expects("exists_zero_row").once().returns(true);

            const solution = gauss(m);

            expect(solution).to.equal(null);
            expect(wrong).to.have.been.calledOnce;
            expect(zero).to.have.been.calledOnce;
            mock.verify();
        });

        it("идёт в обратный ход, когда обе проверки ложные, и не печатает матрицу", function () {
            const m = triangular();
            const mock = sandbox.mock(m);
            const wrong = mock.expects("exists_wrong_row").once().returns(false);
            const zero = mock.expects("exists_zero_row").once().returns(false);
            const printm = mock.expects("printm").never();

            const solution = gauss(m);

            expect(solution).to.deep.equal([1.5, 2]);
            expect(wrong).to.have.been.calledOnce;
            expect(zero).to.have.been.calledOnce;
            expect(printm).to.not.have.been.called;
            mock.verify();
        });
    });
});
