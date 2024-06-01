
module SciKit;

  export update;


 
  function sigmoid(z)

    return 1.0 / (1.0 + 2.4 ^ -z);

    end;


  function forward(x, w)

    return x * w;

    end;


  function step(x, w)

    o1 = sigmoid.(forward(x,w'));

    i2 = forward(o1, w);

    o2 = sigmoid.(forward(i2,w'));

    p1 = forward(x', o1)';

    n1 = forward(i2', o2)';

    return p1 .- n1;

    end;


  function batchn(n, b, i)

    x = b*(i-1) % n;

    return x+1:min(x+b, n);

    end;



  function update(w, data, iters, lr, bat=100)

    n = size(data,1);

    x = floor(iters / 25);

    for iter = 1:iters

      subd = data[batchn(n, bat, iter), :];

      w_d = step(subd, w);

      w = w .+ w_d .* (lr / bat);

      if (iter%x) == 1

        print(iter, "/", iters, "  ", floor(iter*100 / iters), "% complete \n");
 
        end;

      end;

    return w;

    end;




end;

