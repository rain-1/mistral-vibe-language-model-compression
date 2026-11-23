# Linux 3.0 BPE Transformer completions

Trailing whitespace is trimmed for display. Repetition counts use the original generated tokens.

Both checkpoints used the same tokenizer and prompts, seed 3 reset for each prompt, temperature 0.7, top-k 40, and 400 generated BPE tokens per completion.

| Prompt | Previous BPE repeated 4-grams | Full-tree BPE repeated 4-grams |
| --- | ---: | ---: |
| `static int __init` | 350 | 88 |
| `static void` | 16 | 260 |

A repeated 4-gram counts each occurrence after its first among the 397 overlapping four-token windows in a completion.

## `static int __init`

### Previous BPE

````c
static int __init  The pq ($()) {
        |
   ON
   |
   0 |      |
   |
   |
   |
   |
   |
   |
   1 |
   |
   |
   |
   |
   |
   |

   |

   |
   |
   |

   |
   |

   0 |

   |
   |
   |

   |

   |
   |
   |
   |
   |

   |
   |
   |

   |
   |
   |


   |
   |
   |

   |
   |
   |


   |


   |

   |


   |





   |


   ON
   |

   |






   |










   ON

















   ON




   ON









   ON

   >









   |





   |






















   |

























































































































































````

### Full-tree BPE

````c
static int __init  __init init_init(void)
{
	register_driver(&init_timer);
	register_driver(&init_timer);
	register_driver(&init_timer);
	register_driver(&init_timer);
	register_driver(&init_timer);
	register_driver(&init_timer);
	register_driver(&init_timer);

	register_board(&init_timer);
}

static void __exit init_exit(void)
{
	unregister_driver(&init_timer);
}

module_init(init_init);

/*
 * arch/arm/mach-types.c
 *
 * Copyright (C) 2006-2006 Alex Communications Inc.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published
 * published by the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
 *
 * WARRAN
````

## `static void`

### Previous BPE

````c
static void  The media.  The defaults and a task of the CPU
        the CPU.  Each the default is the driver can be used to the array the mutex.  The default value of the data structure, the array.gabilities field of the



The defaults of the number of the
The above has been revised from the array has been is the task of the
the following the array is the current.

	Example: On_ad_mutex_handler_rcu():

	These 1	This attribute that is
	  rcu_mutex(struct kref_map * @_alloc);

	data);

	printk(e_oper.

	(struct *pnp_data);
	...);
	struct padata'.

	struct parport_data, rcu_data);
	/* pop, struct media_entity
	struct parport_mutex_list,
	void *data);
	void (*_data, and will have a function.
	struct media_subsys *buf_sched()
	mutex_lock_sched()

	struct padata_unlock() calls
	struct urb *sgv_count_data_entry_ref, struct media_barrier_entry(p->lock()
	-side->data->ops->data);
	struct list, parport *data, struct kfifo_rcu_data, struct media_data.
	struct parport_lock(&u_lock);

		...
	mutex_lock_read_mutex held_pointer_data_lock(&direction

	data_addr'(* ERROR_tst_handler_each_unlock(void);
	/*
	mutex held a data structure, but is a kset that the following flags a reference *init that
	void structure, but the task structure, and
	struct_torture_entry	lock_data
````

### Full-tree BPE

````c
static void udc_port_pkt_response_ports(struct udc_port *port)
{
	/* Get the port number */
	int port = 0;

	/* Setup port number */
	if (port->port) {
		rc = udc_port_get_port(port);
		if (!rc)
			return;
		return;
	}

	/* Set the port number */
	if (port->port) {
		rc = udc_port_get_port_response_ports(port->port);
		if (!rc)
			return;
	}

	/* Set up port number */
	if (port->port) {
		/* Enable PIO/PCI */
		if (port->port) {
			/* Test the port number */
			int port = 0;

			port->port->port.port->port.port->port.port->port.port->port.port->port.port->port.port->port.port->port.port->port.port->port.port->port.port->port.port->port.port->port.port->port.port->port.port->port.port->port.port->port.port->port.port->port.port->port.port->port.port->port.port->port.port->port.port->port.port->port.port->port.port->port.port->port.port->port.port->port.port->port.port->port.port->port.port->port.port->port.port->port.port->port.port->port.port->port.port->port.port->port.port->port.port->port.port->port.port->port
````
