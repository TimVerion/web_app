# 使用基础镜像库
FROM mayanhui/good:centos7
 
# 创建工作路径
RUN mkdir /app
 
# 指定容器启动时执行的命令都在app目录下执行
WORKDIR /app
 
# 替换nginx的配置
COPY nginx.conf /usr/local/nginx/conf/nginx.conf
 
# 将本地app目录下的内容拷贝到容器的app目录下
COPY ./app/ /app/

# 启动nginx和uwsgi
ENTRYPOINT /usr/local/nginx/sbin/nginx -g "daemon on;" && uwsgi --ini /app/uwsgi.ini
