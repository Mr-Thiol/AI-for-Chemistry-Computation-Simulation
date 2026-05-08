import camelot
import pandas as pd

def extract_si_tables_camelot(pdf_path):
    # Camelot 使用从 1 开始的物理页码
    target_pages = "20-24"
    
    print(f"开始使用 Camelot 解析 {pdf_path} 的第 {target_pages} 页...\n")
    
    # flavor='stream' 更适合没有完整竖线边框的科研表格
    # 如果你的 PDF 表格是完整的 Excel 风格实线网格，可以改为 flavor='lattice'
    tables = camelot.read_pdf(pdf_path, pages=target_pages, flavor='stream')
    
    print(f"共提取到 {tables.n} 个表格结构。\n")
    
    # 目标表格序号，从 Table S3 开始
    current_table_num = 3 
    
    for i, table in enumerate(tables):
        # 提取到 S12 即可停止
        if current_table_num > 12:
            break
            
        # camelot 的 table.df 直接就是一个 pandas DataFrame
        df = table.df
        
        if not df.empty:
            # 第一行通常是表头，我们把它设为列名并删掉原第一行
            df.columns = df.iloc[0]
            df = df[1:]
            
            # 清理：将完全为空字符串的单元格替换为 NaN，然后去掉全空的行和列
            df = df.replace('', pd.NA).dropna(how='all').dropna(axis=1, how='all')
            
            # 导出为 CSV
            output_filename = f"TableS{current_table_num}.csv"
            df.to_csv(output_filename, index=False, encoding="utf-8-sig")
            
            # table.parsing_report 可以查看准确率 (accuracy) 等诊断信息
            accuracy = table.parsing_report['accuracy']
            print(f"✅ 成功提取并保存: {output_filename} (提取自第 {table.page} 页, 准确率评分: {accuracy}%)")
            
            current_table_num += 1

if __name__ == "__main__":
    # 请确保将这里的路径替换为你实际的 PDF 文件路径
    pdf_file_path = "cs8b03077_si_001.pdf" 
    extract_si_tables_camelot(pdf_file_path)
    print("\n所有指定表格提取完毕！")