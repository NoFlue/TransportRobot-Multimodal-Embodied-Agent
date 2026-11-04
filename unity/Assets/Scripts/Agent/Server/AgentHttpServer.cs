using System;
using System.IO;
using System.Net;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace MultimodalAgent
{
    public readonly struct HttpReply
    {
        public readonly int Status;
        public readonly string Json;

        public HttpReply(int status, string json)
        {
            Status = status;
            Json = json;
        }
    }

    /// <summary>
    /// 127.0.0.1에서 JSON 요청을 받는 작은 HTTP 서버. 요청은 스레드 풀에서 처리한다.
    /// Unity API가 필요한 작업은 handler가 메인 스레드로 넘긴다.
    /// </summary>
    public sealed class AgentHttpServer : IDisposable
    {
        public delegate Task<HttpReply> Handler(string method, string path, string body);

        readonly Handler handler;
        HttpListener listener;
        Thread acceptThread;

        public AgentHttpServer(Handler handler) => this.handler = handler;

        public void Start(int port)
        {
            listener = new HttpListener();
            listener.Prefixes.Add($"http://127.0.0.1:{port}/");
            listener.Start();
            acceptThread = new Thread(AcceptLoop) { IsBackground = true, Name = "AgentHttpServer" };
            acceptThread.Start();
        }

        void AcceptLoop()
        {
            while (listener != null && listener.IsListening)
            {
                HttpListenerContext context;
                try
                {
                    context = listener.GetContext();
                }
                catch (Exception)
                {
                    break;
                }
                Task.Run(() => HandleAsync(context));
            }
        }

        async Task HandleAsync(HttpListenerContext context)
        {
            HttpReply reply;
            try
            {
                string body;
                using (var reader = new StreamReader(context.Request.InputStream, Encoding.UTF8))
                {
                    body = await reader.ReadToEndAsync();
                }
                var path = context.Request.Url.AbsolutePath.TrimEnd('/');
                reply = await handler(context.Request.HttpMethod, path.Length == 0 ? "/" : path, body);
            }
            catch (Exception e)
            {
                Debug.LogException(e);
                reply = new HttpReply(500, "{\"error\":" + Newtonsoft.Json.JsonConvert.ToString(e.Message) + "}");
            }

            try
            {
                var bytes = Encoding.UTF8.GetBytes(reply.Json ?? "{}");
                context.Response.StatusCode = reply.Status;
                context.Response.ContentType = "application/json; charset=utf-8";
                context.Response.ContentLength64 = bytes.Length;
                await context.Response.OutputStream.WriteAsync(bytes, 0, bytes.Length);
                context.Response.Close();
            }
            catch (Exception)
            {
                // 클라이언트가 먼저 끊은 경우. 다음 요청 처리에는 영향이 없다.
            }
        }

        public void Dispose()
        {
            var current = listener;
            listener = null;
            if (current == null) return;
            try
            {
                current.Stop();
                current.Close();
            }
            catch (Exception)
            {
                // 이미 닫힌 경우.
            }
        }
    }
}
