from flask import render_template, request, jsonify
from ex import *
from flask import Flask
from config import SQLALCHEMY_DATABASE_URI
from models import db, Property
import csv
from io import TextIOWrapper
from sqlalchemy import text
import base64

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI
db.init_app(app)
with app.app_context():
    db.create_all()
    property_data = {
        'PropType': 0,
        'nbhd': 'aaa',
        'Style': 0,
        'Stories': 'aaa',
        'Year_Built': 0,
        'Rooms': 0,
        'FinishedSqft': 0,
        'Units': 0,
        'Bdrms': 0,
        'Fbath': 0,
        'Hbath': 0,
        'Sale_price': 0,
        'RICH': 'aaa',
        'index': 0
    }
    new_property = Property(
        PropType=property_data['PropType'],
        nbhd=property_data['nbhd'],
        Style=property_data['Style'],
        Stories=property_data['Stories'],
        Year_Built=property_data['Year_Built'],
        Rooms=property_data['Rooms'],
        FinishedSqft=property_data['FinishedSqft'],
        Units=property_data['Units'],
        Bdrms=property_data['Bdrms'],
        Fbath=property_data['Fbath'],
        Hbath=property_data['Hbath'],
        Sale_price=property_data['Sale_price'],
        RICH=property_data['RICH'],
        index=property_data['index']
    )
    db.session.add(new_property)
    db.session.commit()


######################################################################################
# url
@app.route('/get_table_data')
def get_table_data():
    table_name = request.args.get('table')  # 获取前端传递的表名参数
    index = int(table_name[-1])
    query = text(f"SELECT * FROM 'properties' WHERE [index] = {index} LIMIT 50")
    table_data = db.session.execute(query).fetchall()
    # 列名定义
    column_names = [
        'PropertyID', 'PropType', 'nbhd', 'Style', 'Stories',
        'Year_Built', 'Rooms', 'FinishedSqft', 'Units', 'Bdrms',
        'Fbath', 'Hbath', 'Sale_price', 'RICH'
    ]

    # 转换RowProxy对象为字典列表
    result_list = [dict(zip(column_names, row)) for row in table_data]

    # 将数据以 JSON 格式返回给前端
    return jsonify(result_list)


@app.route('/table_data')
def table_data():
    table_name = request.args.get('table')  # 获取前端传递的表名参数
    index = int(table_name[-1])
    query = text(f"SELECT * FROM 'properties' WHERE [index] = {index} ")
    table_data = db.session.execute(query).fetchall()
    # 列名定义
    column_names = [
        'PropertyID', 'PropType', 'nbhd', 'Style', 'Stories',
        'Year_Built', 'Rooms', 'FinishedSqft', 'Units', 'Bdrms',
        'Fbath', 'Hbath', 'Sale_price', 'RICH'
    ]

    # 转换RowProxy对象为字典列表
    result_list = [dict(zip(column_names, row)) for row in table_data]

    # 将数据以 JSON 格式返回给前端
    return jsonify(result_list)


@app.route('/fetch_table_names')
def fetch_table_names():
    try:
        query = text(f"SELECT MAX([index]) FROM {'properties'} LIMIT 1")
        table_data = db.session.execute(query).fetchall()
        table_names = []
        for i in range(1, table_data[0][0] + 1):
            table_names.append('File' + str(i))
        return jsonify(table_names)
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)})


# 新增路由来处理文件上传
@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files['file']
        if file:
            # 使用 TextIOWrapper 处理文件编码
            csv_reader = csv.reader(TextIOWrapper(file, encoding='utf-8'))
            query = text(f"SELECT MAX([index]) FROM {'properties'} LIMIT 1")
            table_data = db.session.execute(query).fetchall()
            print(table_data)
            # 处理上传的 CSV 文件
            for row in csv_reader:
                property_data = {
                    'PropType': row[1],
                    'nbhd': row[2],
                    'Style': row[3],
                    'Stories': row[4],
                    'Year_Built': row[5],
                    'Rooms': row[6],
                    'FinishedSqft': row[7],
                    'Units': row[8],
                    'Bdrms': row[9],
                    'Fbath': row[10],
                    'Hbath': row[11],
                    'Sale_price': row[12],
                    'RICH': row[13],
                    'index': table_data[0][0] + 1
                }

                property_data = {key: value if value != 'NULL' else None for key, value in property_data.items()}
                property_data = {key: value if value != '' else None for key, value in property_data.items()}

                new_property = Property(
                    PropType=property_data['PropType'],
                    nbhd=property_data['nbhd'],
                    Style=property_data['Style'],
                    Stories=property_data['Stories'],
                    Year_Built=property_data['Year_Built'],
                    Rooms=property_data['Rooms'],
                    FinishedSqft=property_data['FinishedSqft'],
                    Units=property_data['Units'],
                    Bdrms=property_data['Bdrms'],
                    Fbath=property_data['Fbath'],
                    Hbath=property_data['Hbath'],
                    Sale_price=property_data['Sale_price'],
                    RICH=property_data['RICH'],
                    index=property_data['index']
                )
                db.session.add(new_property)
            db.session.commit()
            # print("Data committed to the database successfully.")
            # 返回上传成功的消息
            return jsonify({'message': 'File uploaded successfully'})
        else:
            return jsonify({'error': 'No file provided'})
    except Exception as e:
        print("捕获到异常：", str(e))
        return jsonify({'error': f'Error uploading file: {str(e)}'})


# 当访问根路径时渲染 index.html 模板
@app.route('/')
def index():
    return render_template('index.html')


# 算法路由
@app.route('/run_algorithm')
def run_algorithm():
    algorithm_name = request.args.get('name')  # 获取算法名称参数
    table_name = request.args.get('table')  # 获取前端传递的表名参数

    # 根据算法名称调用相应的算法函数
    if algorithm_name == 'dbscan':
        result = dbscan(table_name)
    elif algorithm_name == 'km':
        result = km(table_name)
    elif algorithm_name == 'kme':
        result = kme(table_name)
    else:
        result = {'error': 'Invalid algorithm name'}

    if 'image_base64' in result:
        # 将Base64字符串返回给前端
        return jsonify({'image_base64': result['image_base64'], 'result_text': result['result_text']})
    else:
        # 将其他结果作为JSON返回
        return jsonify(result)


######################################################################################
# main
# 在主程序中运行 Flask 应用
if __name__ == '__main__':
    app.run()
