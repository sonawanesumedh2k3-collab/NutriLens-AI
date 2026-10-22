# ============================================================
# NutriLens AI
# Packaged Food Nutrition Analyzer
#
# Flask + Machine Learning + MySQL
#
# This application:
# 1. Accepts packaged food information
# 2. Converts nutrition values to per 100g
# 3. Calculates the project health score
# 4. Uses the trained Random Forest model
# 5. Stores the prediction in MySQL
# 6. Displays prediction history
# 7. Displays the dashboard
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path

import joblib
import pandas as pd

from flask import Flask, render_template, request,redirect, url_for

import mysql.connector

from dotenv import load_dotenv
import os

load_dotenv()


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# MYSQL DATABASE CONFIGURATION
# ============================================================

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "nutritional_health_db")
}


# ============================================================
# MACHINE LEARNING MODEL
# ============================================================

MODEL_PATH = BASE_DIR / "model" / "best_model.pkl"

SCALER_PATH = BASE_DIR / "model" / "scaler.pkl"


# Load the trained Random Forest model
model = joblib.load(MODEL_PATH)


# Load the StandardScaler used during model training
scaler = joblib.load(SCALER_PATH)


# ============================================================
# DATABASE CONNECTION FUNCTION
# ============================================================

def get_db_connection():

    """
    Creates and returns a connection to MySQL.
    """

    connection = mysql.connector.connect(
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        database=DB_CONFIG["database"]
    )

    return connection


# ============================================================
# HEALTH SCORE FUNCTIONS
# ============================================================

def calorie_points(calories):

    if calories <= 80:
        return 0

    elif calories <= 160:
        return 1

    elif calories <= 240:
        return 2

    elif calories <= 320:
        return 3

    elif calories <= 400:
        return 4

    elif calories <= 480:
        return 5

    elif calories <= 560:
        return 6

    elif calories <= 640:
        return 7

    elif calories <= 720:
        return 8

    elif calories <= 800:
        return 9

    else:
        return 10


# ------------------------------------------------------------


def sugar_points(sugar):

    if sugar <= 4.5:
        return 0

    elif sugar <= 9:
        return 1

    elif sugar <= 13.5:
        return 2

    elif sugar <= 18:
        return 3

    elif sugar <= 22.5:
        return 4

    elif sugar <= 27:
        return 5

    elif sugar <= 31.5:
        return 6

    elif sugar <= 36:
        return 7

    elif sugar <= 40.5:
        return 8

    elif sugar <= 45:
        return 9

    else:
        return 10


# ------------------------------------------------------------


def saturated_fat_points(saturated_fat):

    if saturated_fat <= 1:
        return 0

    elif saturated_fat <= 2:
        return 1

    elif saturated_fat <= 3:
        return 2

    elif saturated_fat <= 4:
        return 3

    elif saturated_fat <= 5:
        return 4

    elif saturated_fat <= 6:
        return 5

    elif saturated_fat <= 7:
        return 6

    elif saturated_fat <= 8:
        return 7

    elif saturated_fat <= 9:
        return 8

    elif saturated_fat <= 10:
        return 9

    else:
        return 10


# ------------------------------------------------------------


def sodium_points(sodium):

    if sodium <= 90:
        return 0

    elif sodium <= 180:
        return 1

    elif sodium <= 270:
        return 2

    elif sodium <= 360:
        return 3

    elif sodium <= 450:
        return 4

    elif sodium <= 540:
        return 5

    elif sodium <= 630:
        return 6

    elif sodium <= 720:
        return 7

    elif sodium <= 810:
        return 8

    elif sodium <= 900:
        return 9

    else:
        return 10


# ------------------------------------------------------------


def protein_points(protein):

    if protein <= 1.6:
        return 0

    elif protein <= 3.2:
        return 1

    elif protein <= 4.8:
        return 2

    elif protein <= 6.4:
        return 3

    elif protein <= 8.0:
        return 4

    else:
        return 5


# ------------------------------------------------------------


def fiber_points(fiber):

    if fiber <= 0.9:
        return 0

    elif fiber <= 1.8:
        return 1

    elif fiber <= 2.7:
        return 2

    elif fiber <= 3.6:
        return 3

    elif fiber <= 4.5:
        return 4

    else:
        return 5


# ============================================================
# HEALTH SCORE CALCULATION
# ============================================================

def calculate_health_score(
    calories,
    sugar,
    saturated_fat,
    sodium,
    protein,
    fiber
):

    # Negative nutritional factors
    negative_points = (
        calorie_points(calories)
        + sugar_points(sugar)
        + saturated_fat_points(saturated_fat)
        + sodium_points(sodium)
    )

    # Positive nutritional factors
    positive_points = (
        protein_points(protein)
        + fiber_points(fiber)
    )

    # Final project health score
    health_score = negative_points - positive_points

    return health_score


# ============================================================
# HEALTH CATEGORY
# ============================================================

def health_category(score):

    if score <= 5:

        return "Healthy"

    elif score <= 18:

        return "Moderate"

    else:

        return "Unhealthy"


# ============================================================
# HOME / ANALYZER ROUTE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    error = None

    # --------------------------------------------------------
    # When user submits the form
    # --------------------------------------------------------

    if request.method == "POST":

        try:

            # =================================================
            # PRODUCT INFORMATION
            # =================================================

            product_name = request.form.get(
                "product_name",
                ""
            ).strip()

            brand_name = request.form.get(
                "brand_name",
                ""
            ).strip()

            category = request.form.get(
                "category",
                ""
            ).strip()


            # =================================================
            # CHECK PRODUCT NAME
            # =================================================

            if not product_name:

                error = "Please enter the product name."

                return render_template(
                    "index.html",
                    error=error
                )


            # =================================================
            # GET NUTRITION VALUES
            # =================================================

            serving_size = float(
                request.form.get("serving_size")
            )

            calories = float(
                request.form.get("calories")
            )

            sugar = float(
                request.form.get("sugar")
            )

            saturated_fat = float(
                request.form.get("saturated_fat")
            )

            sodium = float(
                request.form.get("sodium")
            )

            protein = float(
                request.form.get("protein")
            )

            fiber = float(
                request.form.get("fiber")
            )


            # =================================================
            # VALIDATION
            # =================================================

            if serving_size <= 0:

                error = "Serving size must be greater than 0."

                return render_template(
                    "index.html",
                    error=error
                )


            if calories < 0:
                error = "Calories cannot be negative."

                return render_template(
                    "index.html",
                    error=error
                )


            if sugar < 0:
                error = "Sugar cannot be negative."

                return render_template(
                    "index.html",
                    error=error
                )


            if saturated_fat < 0:
                error = "Saturated fat cannot be negative."

                return render_template(
                    "index.html",
                    error=error
                )


            if sodium < 0:
                error = "Sodium cannot be negative."

                return render_template(
                    "index.html",
                    error=error
                )


            if protein < 0:
                error = "Protein cannot be negative."

                return render_template(
                    "index.html",
                    error=error
                )


            if fiber < 0:
                error = "Dietary fiber cannot be negative."

                return render_template(
                    "index.html",
                    error=error
                )


            # =================================================
            # CONVERT VALUES TO PER 100g
            # =================================================

            calories_per_100g = (
                calories / serving_size
            ) * 100

            sugar_per_100g = (
                sugar / serving_size
            ) * 100

            saturated_fat_per_100g = (
                saturated_fat / serving_size
            ) * 100

            sodium_per_100g = (
                sodium / serving_size
            ) * 100

            protein_per_100g = (
                protein / serving_size
            ) * 100

            fiber_per_100g = (
                fiber / serving_size
            ) * 100


            # =================================================
            # CREATE DATAFRAME FOR MACHINE LEARNING
            # =================================================

            input_data = pd.DataFrame(
                [
                    {
                        "calories_per_100g":
                            calories_per_100g,

                        "sugar_per_100g":
                            sugar_per_100g,

                        "saturated_fat_per_100g":
                            saturated_fat_per_100g,

                        "sodium_per_100g":
                            sodium_per_100g,

                        "proteins_per_100g":
                            protein_per_100g,

                        "dietary_fiber_per_100g":
                            fiber_per_100g
                    }
                ]
            )


            # =================================================
            # SCALE INPUT DATA
            # =================================================

            scaled_input = scaler.transform(
                input_data
            )


            # =================================================
            # MACHINE LEARNING PREDICTION
            # =================================================

            prediction = model.predict(
                scaled_input
            )[0]


            # Convert prediction to normal Python string
            prediction = str(prediction)


            # =================================================
            # PROJECT HEALTH SCORE
            # =================================================

            health_score = calculate_health_score(

                calories_per_100g,

                sugar_per_100g,

                saturated_fat_per_100g,

                sodium_per_100g,

                protein_per_100g,

                fiber_per_100g
            )


            # =================================================
            # SCORE CATEGORY
            # =================================================

            score_category = health_category(
                health_score
            )


            # =================================================
            # STORE RESULT IN MYSQL
            # =================================================

            connection = None
            cursor = None

            try:

                connection = get_db_connection()

                cursor = connection.cursor()


                # ------------------------------------------------
                # Insert product + nutrition + prediction
                # ------------------------------------------------

                insert_query = """
                    INSERT INTO predictions (

                        product_name,
                        brand_name,
                        category,

                        serving_size_g,

                        calories,
                        sugar,
                        saturated_fat,
                        sodium,
                        protein,
                        dietary_fiber,

                        calories_per_100g,
                        sugar_per_100g,
                        saturated_fat_per_100g,
                        sodium_per_100g,
                        protein_per_100g,
                        fiber_per_100g,

                        health_score,
                        score_category,
                        ml_prediction

                    )

                    VALUES (

                        %s,
                        %s,
                        %s,

                        %s,

                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,

                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,

                        %s,
                        %s,
                        %s

                    )
                """


                values = (

                    product_name,

                    brand_name,

                    category,

                    serving_size,

                    calories,

                    sugar,

                    saturated_fat,

                    sodium,

                    protein,

                    fiber,

                    calories_per_100g,

                    sugar_per_100g,

                    saturated_fat_per_100g,

                    sodium_per_100g,

                    protein_per_100g,

                    fiber_per_100g,

                    health_score,

                    score_category,

                    prediction
                )


                cursor.execute(
                    insert_query,
                    values
                )


                connection.commit()


                print(
                    "Prediction successfully saved to MySQL."
                )


            except Exception as db_error:

                print(
                    "Database insertion error:"
                )

                print(db_error)


                error = (
                    "Prediction was generated, "
                    "but it could not be saved to MySQL."
                )


            finally:

                if cursor is not None:

                    cursor.close()


                if (
                    connection is not None
                    and connection.is_connected()
                ):

                    connection.close()


            # =================================================
            # RESULT OBJECT
            # =================================================

            result = {

                "prediction": prediction,

                "score": health_score,

                "score_category": score_category,

                "product_name": product_name,

                "brand_name": brand_name,

                "category": category,

                "calories_per_100g":
                    calories_per_100g,

                "sugar_per_100g":
                    sugar_per_100g,

                "saturated_fat_per_100g":
                    saturated_fat_per_100g,

                "sodium_per_100g":
                    sodium_per_100g,

                "protein_per_100g":
                    protein_per_100g,

                "fiber_per_100g":
                    fiber_per_100g
            }


        # =====================================================
        # HANDLE INVALID INPUT
        # =====================================================

        except ValueError:

            error = (
                "Please enter valid numeric values "
                "for all nutrition fields."
            )


        # =====================================================
        # HANDLE GENERAL ERROR
        # =====================================================

        except Exception as e:

            print(
                "Application error:"
            )

            print(e)

            error = (
                "Something went wrong while processing "
                "the prediction."
            )


    # =========================================================
    # RENDER ANALYZER PAGE
    # =========================================================

    return render_template(

        "index.html",

        result=result,

        error=error
    )


# ============================================================
# PREDICTION HISTORY
# ============================================================

@app.route("/history")
def history():

    predictions = []

    connection = None
    cursor = None

    try:

        connection = get_db_connection()

        cursor = connection.cursor(
            dictionary=True
        )


        # ----------------------------------------------------
        # Get all predictions
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT
                id,
                product_name,
                brand_name,
                category,
                serving_size_g,
                calories,
                sugar,
                saturated_fat,
                sodium,
                protein,
                dietary_fiber,
                health_score,
                score_category,
                ml_prediction,
                created_at

            FROM predictions

            ORDER BY id DESC
            """
        )


        predictions = cursor.fetchall()


    except Exception as e:

        print(
            "History database error:"
        )

        print(e)


    finally:

        if cursor is not None:

            cursor.close()


        if (
            connection is not None
            and connection.is_connected()
        ):

            connection.close()


    return render_template(

        "history.html",

        predictions=predictions
    )

# =========================================================
# PRODUCT COMPARISON
# =========================================================

@app.route("/compare")
def compare():

    # Get selected product IDs from the URL
    selected_ids = request.args.getlist("ids")

    # Make sure exactly 2 products are selected
    if len(selected_ids) != 2:
        return redirect(url_for("history"))

    try:
        # Convert IDs to integers
        product_id_1 = int(selected_ids[0])
        product_id_2 = int(selected_ids[1])

        conn = mysql.connector.connect(**DB_CONFIG)

        cursor = conn.cursor(dictionary=True)

        cursor.execute("""
            SELECT
                id,
                product_name,
                brand_name,
                category,
                serving_size_g,
                calories,
                sugar,
                saturated_fat,
                sodium,
                protein,
                dietary_fiber,
                health_score,
                score_category,
                ml_prediction,
                created_at
            FROM predictions
            WHERE id IN (%s, %s)
            ORDER BY id
        """, (product_id_1, product_id_2))

        products = cursor.fetchall()

        cursor.close()
        conn.close()

        # Make sure both products exist
        if len(products) != 2:
            return redirect(url_for("history"))

        return render_template(
            "compare.html",
            products=products
        )

    except (ValueError, mysql.connector.Error):

        return redirect(url_for("history"))
# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
def dashboard():

    connection = None
    cursor = None


    # --------------------------------------------------------
    # Default values
    # --------------------------------------------------------

    total_predictions = 0

    healthy_count = 0

    moderate_count = 0

    unhealthy_count = 0

    recent_predictions = []


    try:

        connection = get_db_connection()

        cursor = connection.cursor(
            dictionary=True
        )


        # =====================================================
        # TOTAL PREDICTIONS
        # =====================================================

        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM predictions
            """
        )


        result = cursor.fetchone()


        total_predictions = result["total"]


        # =====================================================
        # CATEGORY COUNTS
        # =====================================================

        cursor.execute(
            """
            SELECT
                score_category,
                COUNT(*) AS total

            FROM predictions

            GROUP BY score_category
            """
        )


        category_results = cursor.fetchall()


        for row in category_results:

            category = row["score_category"]

            count = row["total"]


            if category == "Healthy":

                healthy_count = count


            elif category == "Moderate":

                moderate_count = count


            elif category == "Unhealthy":

                unhealthy_count = count


        # =====================================================
        # RECENT PREDICTIONS
        # =====================================================

        cursor.execute(
            """
            SELECT

                id,

                product_name,

                brand_name,

                category,

                calories,

                sugar,

                health_score,

                score_category,

                ml_prediction,

                created_at

            FROM predictions

            ORDER BY id DESC

            LIMIT 10
            """
        )


        recent_predictions = cursor.fetchall()


    except Exception as e:

        print(
            "Dashboard database error:"
        )

        print(e)


    finally:

        if cursor is not None:

            cursor.close()


        if (
            connection is not None
            and connection.is_connected()
        ):

            connection.close()


    # =========================================================
    # RENDER DASHBOARD
    # =========================================================

    return render_template(

        "dashboard.html",

        total_predictions=total_predictions,

        healthy_count=healthy_count,

        moderate_count=moderate_count,

        unhealthy_count=unhealthy_count,

        recent_predictions=recent_predictions
    )


# =========================================================
# METHODOLOGY PAGE
# =========================================================

@app.route("/methodology")
def methodology():

    return render_template("methodology.html")

# ============================================================
# MYSQL CONNECTION TEST
# ============================================================

def test_mysql_connection():

    connection = None

    try:

        connection = get_db_connection()


        if connection.is_connected():

            print(
                "MySQL connection successful."
            )


    except Exception as e:

        print(
            "MySQL connection failed:"
        )

        print(e)


    finally:

        if (
            connection is not None
            and connection.is_connected()
        ):

            connection.close()


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "=============================================="
    )

    print(
        "        NutriLens AI"
    )

    print(
        "  Packaged Food Nutrition Analyzer"
    )

    print(
        "=============================================="
    )

    print()


    # Test MySQL before starting Flask
    test_mysql_connection()


    print()

    print(
        "Starting Flask application..."
    )

    print(
        "Open: http://127.0.0.1:5000/"
    )

    print()


    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )